#include <torch/torch.h>
#include <cmath>
#include <iostream>

int main() {
    int batch_size = 16;
    int n_head = 12;
    int seq_len = 64;
    int head_embd = 64;

    torch::Tensor q = torch::randn({batch_size, n_head, seq_len, head_embd}, torch::kCUDA);
    torch::Tensor k = torch::randn({batch_size, n_head, seq_len, head_embd}, torch::kCUDA);
    torch::Tensor v = torch::randn({batch_size, n_head, seq_len, head_embd}, torch::kCUDA);

    std::cout << "=== profiling manual attention ===" << std::endl;

    auto manual_attn = [](torch::Tensor q, torch::Tensor k, torch::Tensor v) {
        auto att = (q @ k.transpose(-2, -1)) * (1.0 / std::sqrt(k.size(-1)));
        att = torch::nn::functional::softmax(att, -1);
        auto y = att @ v;
        return y;
    };

    {
        torch::autograd::profiler::Profile prof(true);
        torch::Tensor manual_result = manual_attn(q, k, v);
        // C++中没有直接打印profiler信息的方法，需要手动处理或者使用其他工具
    }

    std::cout << "=== profiling minimal flash attention === " << std::endl;

    {
        torch::autograd::profiler::Profile prof(true);
        // 假设minimal_attn是一个已经定义好的C++函数或者类
        torch::Tensor minimal_result = minimal_attn_forward(q, k, v);
        // 同样，需要手动处理profiler信息
    }

    std::cout << "attn values sanity check: " << torch::allclose(minimal_result, manual_result, 0, 1e-02) << std::endl;

    return 0;
}

