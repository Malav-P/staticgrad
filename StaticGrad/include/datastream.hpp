#ifndef DATASTREAM_HPP
#define DATASTREAM_HPP

#include "node.hpp"
#include <fstream>

class DataStream {
    public:

        std::fstream* stream;
        uint16_t* buffer;
        int buffersize; // in bytes

        explicit DataStream():
            stream(new std::fstream),
            buffer(nullptr),
            buffersize(0)
            {}
        DataStream(const std::string& filepath):
            stream(new std::fstream),
            buffer(nullptr),
            buffersize(0)
            {
                this->open(filepath);
            }

        void init_buffer(int num_tokens);

        void open(const std::string& filePath);
        void close();

        void load_buffer();
        void buffer_to_Node(Node* node, size_t num_tokens);

        ~DataStream(){
            close();
            delete stream;
            delete[] buffer;
        }


};


#endif // DATASTREAM_HPP