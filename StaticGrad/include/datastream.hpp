#ifndef DATASTREAM_HPP
#define DATASTREAM_HPP

#include <memory>
#include <fstream>
#include <iostream>

class DataStream {
    public:

        std::fstream* stream;
        u_int16_t* buffer;
        int buffersize; // in bytes

        explicit DataStream():
            stream(new std::fstream),
            buffer(nullptr),
            buffersize(0)
            {}

        void init_buffer(int num_tokens);

        void open(const std::string& filePath);
        void close();

        void load_buffer();

        ~DataStream(){
            delete stream;
            delete[] buffer;
        }


};

void load_weights(float* dest, const std::string& fname, int expected_bytes = -1);

#endif // DATASTREAM_HPP