#include "../include/datastream.hpp"


void DataStream::open(const std::string& filePath){

    if (!stream->is_open()){
        stream->open(filePath, std::ios::in | std::ios::binary);
        // Check if the file was opened successfully
        if (!stream->is_open()) {
            throw std::runtime_error("error opening binary file\n");
        }
        
    }

    else {
        std::cerr << "WARN: File not read, another file is open. Please use close() to close current file.\n";
    }

}

void DataStream::close(){
    if (stream->is_open()) {
        stream->close();
    }
}

void DataStream::load_buffer(u_int16_t* buffer, int num_bytes){
    if (stream->is_open()){
        stream->read(reinterpret_cast<char*>(buffer), num_bytes);

        if (stream->eof()){
            std::cerr << "WARN: EOF reached. The buffer is not completely loaded.\n";
        }
        else if (stream->fail()) {
            throw std::runtime_error("failed to read into buffer"); 
        }
    }

    else {
        std::cerr << "WARN: Bytes not loaded, no file is open. Please use open(...) to open a file.\n";
    }


}