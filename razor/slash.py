"""
 OCCAM

 Copyright (c) 2011-2017, SRI International

  All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name of SRI International nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import getopt
import sys
import os
import tempfile
import collections

from . import utils

from . import passes

from . import interface

from . import provenance

from . import pool

from . import driver

from . import config

def entrypoint():
    """This is the main entry point

    razor [--work-dir=<dir>] <manifest>

    Previrtualize a compilation unit based on its manifest.

        --work-dir <dir>  : Output intermediate files to the given location <dir>
        --no-strip        : Leave symbol information in the binary
        --devirt          : Devirtualize indirect function calls
        --llpe            : Use Smowton's LLPE for intra-module prunning
        --ipdse           : Apply inter-procedural dead store elimination (experimental)
        --stats           : Show some stats before and after specialization
        --no-specialize   : Do not specialize any intermodule calls
        --tool <tool>     : Print the path to the tool and exit.

    """
    return Slash(sys.argv).run() if utils.checkOccamLib() else 1


def  usage(exe):
    template = '{0} [--work-dir=<dir>]  [--force] [--no-strip] [--devirt] [--llpe] [--ipdse] [--stats] [--no-specialize] <manifest>\n'
    sys.stderr.write(template.format(exe))


class Slash(object):

    def __init__(self, argv):
        utils.setLogger()

        try:
            cmdflags = ['work-dir=',
                        'force',
                        'no-strip',
                        'devirt',
                        'llpe',
                        'ipdse',
                        'stats',
                        'no-specialize',
                        'tool=']
            parsedargs = getopt.getopt(argv[1:], None, cmdflags)
            (self.flags, self.args) = parsedargs

            tool = utils.get_flag(self.flags, 'tool', None)
            if tool is not None:
                tool = config.get_llvm_tool(tool)
                print('tool = {0}'.format(tool))
                sys.exit(0)

        except Exception:
            usage(argv[0])
            self.valid = False
            return

        self.manifest = utils.get_manifest(self.args)
        if self.manifest is None:
            usage(argv[0])
            self.valid = False
            return
        self.work_dir = utils.get_work_dir(self.flags)
        self.pool = pool.getDefaultPool()
        self.valid = True



    def run(self):

        if not self.valid:
            return 1

        if not utils.make_work_dir(self.work_dir):
            return 1

        parsed = utils.check_manifest(self.manifest)

        valid = parsed[0]

        if not valid:
            return 1

        (valid, module, binary, libs, native_libs, ldflags, args, name, constraints) = parsed


        no_strip = utils.get_flag(self.flags, 'no-strip', None)

        devirt = utils.get_flag(self.flags, 'devirt', None)

        use_llpe = utils.get_flag(self.flags, 'llpe', None)

        use_ipdse = utils.get_flag(self.flags, 'ipdse', None)

        show_stats = utils.get_flag(self.flags, 'stats', None)

        no_specialize = utils.get_flag(self.flags, 'no-specialize', None)

        sys.stderr.write('\nslash working on {0} wrt {1} ...\n'.format(module, ' '.join(libs)))

        #<delete this once done>
        #new_libs = []
        #for lib in libs:
        #    new_libs.append(os.path.realpath(lib))
        #
        #libs = new_libs
        #</delete this once done>

        native_lib_flags = []

        #this is simplistic. we are assuming they are (possibly)
        #relative paths, if we need to use a search path then this
        #will need to be beefed up.
        new_native_libs = []
        for lib in native_libs:
            if lib.startswith('-l'):
                native_lib_flags.append(lib)
                continue
            if os.path.exists(lib):
                new_native_libs.append(os.path.realpath(lib))
        native_libs = new_native_libs

        files = utils.populate_work_dir(module, libs, self.work_dir)

        os.chdir(self.work_dir)

        #specialize the arguments ...
        if args is not None:
            main = files[module]
            pre = main.get()
            post = main.new('a')
            passes.specialize_program_args(pre, post, args, 'arguments', name=name)
        elif constraints:
            print('Using the constraints: {0}\n'.format(constraints))
            main = files[module]
            pre = main.get()
            post = main.new('a')
            passes.constrain_program_args(pre, post, constraints, 'constraints')

        #watches were inserted here ...

        #Resolve indirect calls using a pointer analysis
        def _devirt(m):
            "Devirtualize indirect function calls"
            pre = m.get()
            post = m.new('d')
            passes.devirt(pre, post)

        if devirt is not None:
            pool.InParallel(_devirt, files.values(), self.pool)

        # Internalize everything that we can
        # We can never internalize main
        interface.writeInterface(interface.mainInterface(), 'main.iface')

        # First compute the simple interfaces
        vals = files.items()
        def mkvf(k):
            return provenance.VersionedFile(utils.prevent_collisions(k[:k.rfind('.bc')]),
                                            'iface')
        refs = dict([(k, mkvf(k)) for (k, _) in vals])

        def _references((m, f)):
            "Computing references"
            nm = refs[m].new()
            passes.interface(f.get(), nm, [])

        pool.InParallel(_references, vals, self.pool)

        def _internalize((m, i)):
            "Internalizing from references"
            pre = i.get()
            post = i.new('i')
            ifaces = [refs[f].get() for f in refs.keys() if f != m] + ['main.iface']
            passes.internalize(pre, post, ifaces)

        pool.InParallel(_internalize, vals, self.pool)

        # Strip everything
        # This is safe because external names are not stripped
        if no_strip is None:
            def _strip(m):
                "Stripping symbols"
                pre = m.get()
                post = m.new('x')
                passes.strip(pre, post)

            pool.InParallel(_strip, files.values(), self.pool)

        profile_map_before = collections.OrderedDict()
        profile_map_after = collections.OrderedDict()
        # Begin main loop
        iface_before_file = provenance.VersionedFile('interface_before', 'iface')
        iface_after_file = provenance.VersionedFile('interface_after', 'iface')
        progress = True
        rewrite_files = {}
        for m, _ in files.iteritems():
            base = utils.prevent_collisions(m[:m.rfind('.bc')])
            rewrite_files[m] = provenance.VersionedFile(base, 'rw')
        iteration = 0

        if show_stats is not None:
            for m in files.values():
                _, name = tempfile.mkstemp()
                profile_map_before[m.get()] = name
            def _profile_before(m):
                #sys.stderr.write("Profiling %s and storing in %s...\n" % (m.get(), profile_map_before[m.get()]))
                passes.profile(m.get(), profile_map_before[m.get()])
            pool.InParallel(_profile_before, files.values(), self.pool)

        while progress:

            iteration += 1
            progress = False

            # resolve indirect calls using a pointer analysis
            #if devirt is not None:
            #    pool.InParallel(_devirt, files.values(), self.pool)

            # Intra-module previrt
            def intra(m):
                "Intra-module previrtualization"
                # for m in files.values():
                # intra-module previrt
                pre = m.get()
                pre_base = os.path.basename(pre)
                post = m.new('p')
                post_base = os.path.basename(post)
                fn = 'previrt_%s-%s' % (pre_base, post_base)
                passes.peval(pre, post, use_llpe, use_ipdse, log=open(fn, 'w'))

            pool.InParallel(intra, files.values(), self.pool)

            # Inter-module previrt
            iface = passes.deep([x.get() for x in files.values()],
                                ['main.iface'])
            interface.writeInterface(iface, iface_before_file.new())

            if no_specialize is None:
                # Specialize
                def _spec((nm, m)):
                    "Inter-module module specialization"
                    pre = m.get()
                    post = m.new('s')
                    rw = rewrite_files[nm].new()
                    passes.specialize(pre, post, rw, [iface_before_file.get()])

                pool.InParallel(_spec, files.items(), self.pool)

            # Rewrite
            def rewrite((nm, m)):
                "Inter-module module rewriting"
                pre = m.get()
                post = m.new('r')
                rws = [rewrite_files[x].get() for x in files.keys()
                       if x != nm]
                out = [None]
                retcode = passes.rewrite(pre, post, rws, output=out)
                fn = 'rewrite_%s-%s' % (os.path.basename(pre),
                                        os.path.basename(post))
                dbg = open(fn, 'w')
                dbg.write(out[0])
                dbg.close()
                return retcode

            rws = pool.InParallel(rewrite, files.items(), self.pool)
            progress = any(rws)

            # Aggressive internalization
            pool.InParallel(_references, vals, self.pool)
            pool.InParallel(_internalize, vals, self.pool)

            # Compute the interface again
            iface = passes.deep([x.get() for x in files.values()], ['main.iface'])
            interface.writeInterface(iface, iface_after_file.new())

            # Prune
            def prune(m):
                "Pruning dead code/variables"
                pre = m.get()
                post = m.new('occam')
                passes.internalize(pre, post, [iface_after_file.get()])
            pool.InParallel(prune, files.values(), self.pool)

        if show_stats is not None:
            for m in files.values():
                _, name = tempfile.mkstemp()
                profile_map_after[m.get()] = name
            def _profile_after(m):
                #sys.stderr.write("Profiling %s and storing in %s...\n" % (m.get(), profile_map_after[m.get()]))
                passes.profile(m.get(), profile_map_after[m.get()])
            pool.InParallel(_profile_after, files.values(), self.pool)

        # Make symlinks for the "final" versions
        for x in files.values():
            trg = x.base('-final')
            if os.path.exists(trg):
                os.unlink(trg)
            os.symlink(x.get(), trg)


        final_libs = [files[x].get() for x in libs]
        final_module = files[module].get()

        linker_args = final_libs + native_libs + native_lib_flags + ldflags

        link_cmd = '\nclang++ {0} -o {1} {2}\n'.format(final_module, binary, ' '.join(linker_args))

        sys.stderr.write('\nLinking ...\n')
        sys.stderr.write(link_cmd)
        driver.linker(final_module, binary, linker_args)
        sys.stderr.write('\ndone.\n')
        pool.shutdownDefaultPool()

        if show_stats is not None:
            for (f1,v1), (f2,v2) in zip(profile_map_before.iteritems(), \
                                        profile_map_after.iteritems()) :
                sys.stderr.write(f1)
                sys.stderr.write(f2)

                def _splitext(abspath):
                    #Given abspath of the form basename.ext1.ext2....extn
                    #return basename.ext1
                    base = os.path.basename(abspath)
                    res = base.split(os.extsep)
                    assert(len(res) > 1)
                    return res[0] + '.' + res[1]

                f1 = _splitext(f1)
                f2 = _splitext(f2)
                assert (f1 == f2)

                sys.stderr.write('\n\nStatistics for %s before specialization\n' % f1)
                fd = open(v1, 'r')
                for line in fd:
                    sys.stderr.write('\t')
                    sys.stderr.write(line)
                fd.close()
                os.remove(v1)
                sys.stderr.write('Statistics for %s after specialization\n' % f2)
                fd = open(v2, 'r')
                for line in fd:
                    sys.stderr.write('\t')
                    sys.stderr.write(line)
                fd.close()
                os.remove(v2)

        return 0
