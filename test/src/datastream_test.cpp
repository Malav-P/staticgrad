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


TEST_F(DataStreamTest, createDataStream) {


    EXPECT_TRUE(ds->stream != nullptr);

    EXPECT_TRUE(ds != nullptr);
    EXPECT_TRUE(ds->stream->is_open() == false);


}

TEST_F(DataStreamTest, openfile) {

    std::string filepath = "/Users/malavpatel/Coding_Projects/StaticGrad/tokenizer/tokens/tinystories.bin";

    EXPECT_NO_THROW(ds->open(filepath));
    EXPECT_TRUE(ds->stream != nullptr);
    EXPECT_TRUE(ds->stream->is_open() == true);

}

TEST_F(DataStreamTest, openinvalidfile) {

    std::string filepath = "/invalid/file/path";

    EXPECT_THROW(ds->open(filepath), std::runtime_error);

}

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

TEST_F(DataStreamTest, load_weights) {

    std::string filepath = "/Users/malavpatel/Coding_Projects/StaticGrad/models/gpt2.bin";

    int num_params = 124439808;
    auto params = new float[num_params]; // num parameters in gpt2 is 124439808


    EXPECT_NO_THROW(load_weights(params, filepath));

    int wrong_expected_bytes = 10;
    EXPECT_THROW(load_weights(params, filepath, wrong_expected_bytes), std::runtime_error);

    int correct_expected_bytes = num_params*sizeof(float);
    EXPECT_NO_THROW(load_weights(params, filepath, correct_expected_bytes));

    delete[] params;

}




// Main function to run the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}