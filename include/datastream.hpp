#ifndef DATASTREAM_HPP
#define DATASTREAM_HPP

#include <memory>
#include <fstream>
#include <iostream>

class DataStream {
    public:

        std::fstream* stream;

        explicit DataStream():
            stream(new std::fstream)
            {}

        void open(const std::string& filePath);
        void close();

        void load_buffer(uint16_t* buffer, int num_bytes);

        ~DataStream(){
            delete stream;
        }


};


#endif // DATASTREAM_HPP