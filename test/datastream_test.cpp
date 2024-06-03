#include <gtest/gtest.h>
#include "include/datastream.hpp"

class DataStreamTest : public ::testing::Test {
    protected:
    void SetUp() override {}

    void TearDown() override {}

};


TEST_F(DataStreamTest, createDataStream) {

    DataStream* ds = nullptr;

    EXPECT_NO_THROW(ds = new DataStream());
    EXPECT_TRUE(ds->stream != nullptr);

    EXPECT_TRUE(ds != nullptr);
    EXPECT_TRUE(ds->stream->is_open() == false);

    delete ds;

}

TEST_F(DataStreamTest, openfile) {

    DataStream* ds = new DataStream();

    std::string filepath = "/Users/malavpatel/Coding_Projects/StaticGrad/tokenizer/tokens/tinystories.bin";

    EXPECT_NO_THROW(ds->open(filepath));
    EXPECT_TRUE(ds->stream != nullptr);
    EXPECT_TRUE(ds->stream->is_open() == true);

    delete ds;

}

TEST_F(DataStreamTest, openinvalidfile) {

    DataStream* ds = new DataStream();

    std::string filepath = "/invalid/file/path";

    EXPECT_THROW(ds->open(filepath), std::runtime_error);

}

TEST_F(DataStreamTest, loadbuffer) {

    DataStream* ds = new DataStream();
    std::string filepath = "/Users/malavpatel/Coding_Projects/StaticGrad/tokenizer/tokens/tinystories.bin";
    ds->open(filepath);

    int BUFFERSIZE = 32;
    uint16_t* buffer = new uint16_t[BUFFERSIZE];

    EXPECT_NO_THROW(ds->load_buffer(buffer, BUFFERSIZE*sizeof(uint16_t)));
    EXPECT_EQ(buffer[0], 50256); // first token is gpt2 eot token

    // for (int i = 0; i < BUFFERSIZE; i++){
    //     std::cout << buffer[i] << ", ";
    // }
    // std::cout << "\n";
    

    delete ds;
    delete[] buffer;

}




// Main function to run the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}