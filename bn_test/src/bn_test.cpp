#include "tf_utils.hpp"
#include <scope_guard.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <dirent.h>
#include <string>
#include <fstream>
#define IMG_SIZE 32

using namespace std;

int main() {
    vector<vector<float>> data_set;
    vector<int> labels;
    vector<TF_Tensor*> in_tensors;
    vector<TF_Output> in_ops;
    const vector<int64_t> input_data_dims = {1, IMG_SIZE, IMG_SIZE, 3};
    const vector<int64_t> input_bool_dims = {1};
    bool is_train;


    data_set.emplace_back(vector<float>(IMG_SIZE*IMG_SIZE*3, 0));
    labels.emplace_back(1);
    cout<<"total samples "<<data_set.size()<<endl;

    auto graph = tf_utils::LoadGraph("bn_model.pb");
    SCOPE_EXIT{ tf_utils::DeleteGraph(graph); }; // Auto-delete on scope exit.
    if (graph == nullptr) {
    cout << "Can't load graph" << endl;
    return 1;
    }

    auto input_data = TF_Output{TF_GraphOperationByName(graph, "Placeholder"), 0};
    auto input_bool = TF_Output{TF_GraphOperationByName(graph, "Placeholder_3"), 0};
    if (input_data.oper == nullptr || input_bool.oper == nullptr) {
        cout << "Can't init input_data" << endl;
        return 2;
    }
    in_ops.emplace_back(input_data);
    in_ops.emplace_back(input_bool);

    auto input_data_tensor = tf_utils::CreateTensor(TF_FLOAT, input_data_dims, data_set[0]);
    auto input_bool_tensor = tf_utils::CreateTensor(TF_BOOL, input_bool_dims.data(), input_bool_dims.size(), &is_train, sizeof(bool));
    SCOPE_EXIT{ tf_utils::DeleteTensor(input_data_tensor); }; // Auto-delete on scope exit.
    SCOPE_EXIT{ tf_utils::DeleteTensor(input_bool_tensor); }; // Auto-delete on scope exit.
    in_tensors.emplace_back(input_data_tensor);
    in_tensors.emplace_back(input_bool_tensor);

    auto out_op = TF_Output{TF_GraphOperationByName(graph, "classification_model/dropout_2/dropout/Identity"), 0};
    if (out_op.oper == nullptr) {
    cout << "Can't init out_op" << endl;
    return 3;
    }

    TF_Tensor* output_tensor = nullptr;
    SCOPE_EXIT{ tf_utils::DeleteTensor(output_tensor); }; // Auto-delete on scope exit.

    auto status = TF_NewStatus();
    SCOPE_EXIT{ TF_DeleteStatus(status); }; // Auto-delete on scope exit.
    auto options = TF_NewSessionOptions();
    uint8_t config[16] ={0x32, 0xe,  0x9, 0x0, 0x0,  0x0, 0x0,  0x0,
                        0x0, 0xe0, 0x3f,0x20, 0x1, 0x2a, 0x1, 0x31}; 
    TF_SetConfig(options, (void*)config, 16, status);
    auto sess = TF_NewSession(graph, options, status);
    TF_DeleteSessionOptions(options);

    if (TF_GetCode(status) != TF_OK) {
    return 4;
    }


    TF_SessionRun(sess,
                nullptr, // Run options.
                in_ops.data(), in_tensors.data(), 2, // Input tensors, input tensor values, number of inputs.
                &out_op, &output_tensor, 1, // Output tensors, output tensor values, number of outputs.
                nullptr, 0, // Target operations, number of targets.
                nullptr, // Run metadata.
                status // Output status.
                );

    if (TF_GetCode(status) != TF_OK) {
    cout << "Error run session "<<TF_GetCode(status)<<"\n";
    return 5;
    }

    // auto data = static_cast<float*>(TF_TensorData(output_tensor));


    TF_CloseSession(sess, status);
    if (TF_GetCode(status) != TF_OK) {
    cout << "Error close session";
    return 6;
    }

    TF_DeleteSession(sess, status);
    if (TF_GetCode(status) != TF_OK) {
    std::cout << "Error delete session";
    return 7;
    }

    return 0;
}
