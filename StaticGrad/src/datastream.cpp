#include "datastream.hpp"

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


void DataStream::buffer_to_Node(Node* node, int num_tokens){
    if (node->size < num_tokens){
        throw std::invalid_argument("node is too small to transfer number of requested tokens");
    }

    if (buffersize / sizeof(u_int16_t) < num_tokens){
        throw std::invalid_argument("buffer is too small to transfer number of requested tokens");
    }

    for (int i = 0; i < num_tokens; i++){
        node->act[i] = buffer[i];
    }
}