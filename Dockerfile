# To compile this with docker use:
# docker build --tag sigpyproc .
# Then to run it:
# docker run --rm -it sigpyproc
# To be able to access local disk on Mac OSX, you need to use Docker for Mac GUI
# and click on 'File sharing', then add your directory, e.g. /data/bl_pks
# Then to run it:
# docker run --rm -it -v /data/bl_pks:/mnt/data sigpyproc
# And if you want to access a port, you need to do a similar thing:
# docker run --rm -it -p 9876:9876 sigpyproc

# INSTALL BASE FROM KERN SUITE
FROM kernsuite/base:3
ARG DEBIAN_FRONTEND=noninteractive

ENV TERM xterm

######
# Do docker apt-installs
RUN docker-apt-install build-essential python-setuptools python-pip python-tk
RUN docker-apt-install git make
RUN docker-apt-install fftw3 fftw3-dev pkg-config
RUN docker-apt-install libomp-dev

#####
# Pip installation of python packages
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install numpy matplotlib ipython tqdm

# Finally, install sigpyproc!
COPY . .
RUN python setup.py install




