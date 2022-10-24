FROM gcc:11.2
LABEL Name=deeppp Version=0.0.1

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get -y update && apt-get -y install
RUN apt-get install -y --no-install-recommends tzdata git build-essential cmake
# RUN wget -qO- "https://cmake.org/files/v3.24/cmake-3.24.0-linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local

COPY . /usr/src/deeppp
WORKDIR /usr/src/deeppp

RUN ./configure.sh
RUN ./build.sh

CMD ["sh", "./test.sh"]
