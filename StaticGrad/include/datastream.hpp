#ifndef DATASTREAM_HPP
#define DATASTREAM_HPP

#include <memory>
#include <fstream>
#include <iostream>

#include "./classes.hpp"

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
        void buffer_to_Node(Node* node, int num_tokens);

        ~DataStream(){
            close();
            delete stream;
            delete[] buffer;
        }


};


#endif // DATASTREAM_HPP