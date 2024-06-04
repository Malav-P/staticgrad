#include "../include/datastream.hpp"

void DataStream::init_buffer(int num_tokens){

    if (buffer != nullptr){
        delete[] buffer;
    }
    
    buffer = new u_int16_t[num_tokens];
    buffersize = num_tokens*sizeof(u_int16_t);
}

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

void DataStream::load_buffer(){
    if (stream->is_open()){
        stream->read(reinterpret_cast<char*>(buffer), buffersize);

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


void load_weights(float* dest, const std::string& fname, int expected_bytes){

    // Open the file in binary mode
    std::ifstream inputFile(fname, std::ios::binary);

    if (!inputFile.is_open()){
        throw std::runtime_error("could not open file");
    }

    // Get the size of the file
    inputFile.seekg(0, std::ios::end);
    std::streamsize fileSize = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    if (expected_bytes != -1){
        if (expected_bytes != fileSize){
            throw std::runtime_error("number of bytes in file does not match expected");
        }
    }

    // Read the contents of the file into the buffer
    if (!inputFile.read(reinterpret_cast<char*>(dest), fileSize)) {
        throw std::runtime_error("error loading contents into buffer");
    }
}