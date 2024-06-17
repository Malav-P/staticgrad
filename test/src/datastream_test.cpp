#include <gtest/gtest.h>
#include "datastream.hpp"

class DataStreamTest : public ::testing::Test {
    protected:
        DataStream* ds;

        void SetUp() override {
            ds = new DataStream();
        }

        void TearDown() override {
            delete ds;
        }

};

/**
 * @brief Test that a DataStream object can be created successfully.
 *
 * This test verifies that the DataStream constructor works correctly and
 * that the object is initialized with a non-null stream pointer.
 */
TEST_F(DataStreamTest, createDataStream) {


    EXPECT_TRUE(ds->stream != nullptr);

    EXPECT_TRUE(ds != nullptr);
    EXPECT_TRUE(ds->stream->is_open() == false);


}

/**
 * @brief Test that a file can be opened successfully using the DataStream class.
 *
 * This test verifies that the DataStream::open function works correctly and
 * that the file is successfully opened. It also checks that the stream pointer
 * is non-null and that the file is open.
 */
TEST_F(DataStreamTest, openfile) {

    std::string filepath = "/Users/malavpatel/Coding_Projects/StaticGrad/tokenizer/tokens/tinystories.bin";

    EXPECT_NO_THROW(ds->open(filepath));
    EXPECT_TRUE(ds->stream != nullptr);
    EXPECT_TRUE(ds->stream->is_open() == true);

}

/**
 * @brief Test that attempting to open an invalid file path throws an exception.
 *
 * This test verifies that the DataStream::open function throws a std::runtime_error
 * when attempting to open a file with an invalid path.
 */
TEST_F(DataStreamTest, openinvalidfile) {

    std::string filepath = "/invalid/file/path";

    EXPECT_THROW(ds->open(filepath), std::runtime_error);

}

/**
 * @brief Test that the DataStream class can load a buffer from a file.
 *
 * This test verifies that the DataStream::load_buffer function works correctly
 * and that the buffer is loaded with data from the file. It also checks that
 * the first token in the buffer is not the GPT-2 EOT token (50256).
 */
TEST_F(DataStreamTest, loadbuffer) {

    std::string filepath = "/Users/malavpatel/Coding_Projects/StaticGrad/tokenizer/tokens/tinystories.bin";
    ds->open(filepath);

    ds->init_buffer(64);

    EXPECT_NO_THROW(ds->load_buffer());
    EXPECT_EQ(ds->buffer[0], 50256); // first token is gpt2 eot token

    // load another batch
    ds->load_buffer();
    EXPECT_NE(ds->buffer[0], 50256); // first token will generally not be the eot token

}


// Main function to run the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}