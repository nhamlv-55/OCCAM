CURLDIR = $(shell pwd)

OS =  $(shell uname)

#LLVMCC=wllvm
#LLVMGET=extract-bc

LLVMCC=gclang
LLVMGET=get-bc

ifeq (Darwin, $(findstring Darwin, ${OS}))
LIBEXT=dylib
else
LIBEXT=so
endif

all: bitcode

CURL=curl-7.64.1
CURLURL=https://curl.haxx.se/download/${CURL}.tar.gz

CURLLIB=libcurl.${LIBEXT}
CURLLIBBC=${CURLLIB}.bc

curlgit:
#	git clone https://github.com/curl/curl.git curlgit
	wget ${CURLURL}; tar xvfz ${CURL}.tar.gz; mv ${CURL} curlgit

# --enable-static
# --disable-debug --disable-dependency-tracking --disable-silent-rules --prefix=${CURLDIR}/install/  --with-darwinssl --without-ca-bundle --without-ca-path
install/bin/curl: curlgit
	cd curlgit; ./buildconf; CC=${LLVMCC} ./configure --prefix=${CURLDIR}/install/
	cd curlgit; make; make install

bitcode: curl.bc ${CURLLIBBC}


curl.bc: install/bin/curl
	${LLVMGET} install/bin/curl
	mv install/bin/curl.bc .

${CURLLIBBC}: install/bin/curl
	${LLVMGET} install/lib/${CURLLIB}
	mv install/lib/*.${LIBEXT}.bc   ./${CURLLIBBC}

clean:
	rm -rf install
	make -C curlgit clean


very_clean: clean
	rm -rf  *.bc curlgit ${CURLDIR}/install
