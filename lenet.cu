/*
 * This code is released into the public domain.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include<pthread.h>
#include "definitions.h"
#define NGPUs 1
#define dev_0 0
#define dev_1 1
//defining thread data for each threads

struct thread_data
{

//	thread_data(){};
	
	float  *t_data, *t_labels, *t_conv1, *t_pool1, *t_conv2, *t_pool2, *t_fc1, *t_fc1relu, *t_fc2, *t_fc2smax, *t_dlossdata, *t_pconv1, *t_pconv1bias, *t_pconv2, *t_pconv2bias, *t_pfc1, *t_pfc1bias, *t_pfc2, *t_pfc2bias, *t_gconv1, *t_gconv1bias, *t_dpool1, *t_gconv2, *t_gconv2bias, *t_dconv2, *t_dpool2, *t_gfc1, *t_gfc1bias, *t_dfc1, *t_dfc1relu, *t_gfc2, *t_gfc2bias, *t_dfc2, *t_onevec;
	
	void *t_workspace;

	//int dev_id;
	TrainingContext& context;	
	ConvBiasLayer& conv1;
       	MaxPoolLayer& pool1;
       	ConvBiasLayer& conv2;
       	MaxPoolLayer& pool2;
	FullyConnectedLayer& fc1;
	FullyConnectedLayer& fc2;
/*	thread_data(TrainingContext& context_g, ConvBiasLayer& conv1_g, MaxPoolLayer& pool1_g, ConvBiasLayer& conv2_g, MaxPoolLayer& pool2_g,
                    FullyConnectedLayer& fc1_g, FullyConnectedLayer& fc2_g, float *d_data, float *d_labels, float *d_conv1,  float *d_pool1,  float *d_conv2,  float *d_pool2, * float d_fc1,  float *d_fc1relu,  float *d_fc2,  float *d_fc2smax,  float *d_dlossdata,  float *d_pconv1,  float *d_pconv1bias,  float *d_pconv2, * float d_pconv2bias, * float d_pfc1, * float d_pfc1bias,  float *d_pfc2,  float *d_pfc2bias,  float *d_gconv1,  float *d_gconv1bias,  float *d_dpool1,  float *d_gconv2,  float *d_gconv2bias,  float *d_dconv2,  float *d_dpool2,  float *d_gfc1,  float *d_gfc1bias,  float *d_dfc1, * float d_dfc1relu, * float d_gfc2, * float d_gfc2bias, * float d_dfc2, * float d_onevec, void *workspace ) : context(context_g), conv1(conv1_g), pool1(pool1_g), conv2(conv2_g), pool2(pool2_g), fc1(fc1_g), fc2(fc2_g){
	}
};*/
	thread_data(TrainingContext& context_g, ConvBiasLayer& conv1_g, MaxPoolLayer& pool1_g, ConvBiasLayer& conv2_g, MaxPoolLayer& pool2_g,
                    FullyConnectedLayer& fc1_g, FullyConnectedLayer& fc2_g, float *d_data, float *d_labels, float *d_conv1,  float *d_pool1,  float *d_conv2,  float *d_pool2, float *d_fc1,  float *d_fc1relu,  float *d_fc2,  float *d_fc2smax,  float *d_dlossdata,  float *d_pconv1,  float *d_pconv1bias,  float *d_pconv2, float *d_pconv2bias,  float *d_pfc1, float *d_pfc1bias,  float *d_pfc2,  float *d_pfc2bias,  float *d_gconv1,  float *d_gconv1bias,  float *d_dpool1,  float *d_gconv2,  float *d_gconv2bias,  float *d_dconv2,  float *d_dpool2,  float *d_gfc1,  float *d_gfc1bias,  float *d_dfc1, float *d_dfc1relu, float *d_gfc2, float *d_gfc2bias, float *d_dfc2, float *d_onevec, void *workspace ) : context(context_g), conv1(conv1_g), pool1(pool1_g), conv2(conv2_g), pool2(pool2_g), fc1(fc1_g), fc2(fc2_g), t_data(d_data), t_labels(d_labels), t_conv1(d_conv1), t_pool1(d_pool1), t_conv2(d_conv2), t_pool2(d_pool2), t_fc1(d_fc1), t_fc1relu(d_fc1relu), t_fc2(d_fc2), t_fc2smax(d_fc2smax), t_dlossdata(d_dlossdata), t_pconv1(d_pconv1), t_pconv1bias(d_pconv1bias), t_pconv2(d_pconv2), t_pconv2bias(d_pconv2bias), t_pfc1(d_pfc1), t_pfc1bias(d_pfc1bias), t_pfc2(d_pfc2), t_pfc2bias(d_pfc2bias), t_gconv1(d_gconv1), t_gconv1bias(d_gconv1bias), t_dpool1(d_dpool1), t_gconv2(d_gconv2), t_gconv2bias(d_gconv2bias), t_dconv2(d_dconv2), t_dpool2(d_dpool2), t_gfc1(d_gfc1), t_gfc1bias(d_gfc1bias), t_dfc1(d_dfc1), t_dfc1relu(d_dfc1relu), t_gfc2(d_gfc2), t_gfc2bias(d_gfc2bias), t_dfc2(d_dfc2), t_onevec(d_onevec), t_workspace(workspace){
	}
};
///////////////////////////////////////////////////////////////////////////////////////////
// Definitions and helper utilities
__global__ void Synch_Gradients(float* N_G0, float* N_G1) {

        int i = threadIdx.x;
        float k = N_G0[i] + N_G1[i];
        N_G0[i] = k;
}

void sync_grad(int size, float *a, float *b) {
        const float alpha = 1.0;
        //const float beta = 0.0;
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSaxpy(handle,size,
                  &alpha,
                  a, 1,
                  b, 1 );

}

void *training_division(void *i)
{

	struct thread_data *d;
	d = (struct thread_data *)i;
	//int id = d->dev_id;
	
//	cudaSetDevice(id);

        // Forward propagation
        d->context.ForwardPropagation(d->t_data, d->t_conv1, d->t_pool1, d->t_conv2, d->t_pool2, d->t_fc1, d->t_fc1relu, d->t_fc2, d->t_fc2smax, d->t_pconv1, d->t_pconv1bias, d->t_pconv2, d->t_pconv2bias, d->t_pfc1, d->t_pfc1bias, d->t_pfc2, d->t_pfc2bias,
                                   d->t_workspace, d->t_onevec);


	//d->context.test(d->d_dlossdata, d->d_fc2smax, id);
        // Backward propagation
       d->context.Backpropagation(d->conv1, d->pool1, d->conv2, d->pool2,
                                d->t_data, d->t_labels, d->t_conv1, d->t_pool1, d->t_conv2, d->t_pool2, d->t_fc1, d->t_fc1relu, d->t_fc2, d->t_fc2smax, d->t_dlossdata, d->t_pconv1, d->t_pconv1bias, d->t_pconv2, d->t_pconv2bias, d->t_pfc1, d->t_pfc1bias, d->t_pfc2, d->t_pfc2bias,
                  d->t_gconv1, d->t_gconv1bias, d->t_dpool1, d->t_gconv2, d->t_gconv2bias, d->t_dconv2, d->t_dpool2, d->t_gfc1, d->t_gfc1bias, d->t_dfc1, d->t_dfc1relu, d->t_gfc2, d->t_gfc2bias, d->t_dfc2, d->t_workspace, d->t_onevec);
}


int main(int argc, char **argv)
{
#ifdef USE_GFLAGS
    gflags::ParseCommandLineFlags(&argc, &argv, true);
#endif

    size_t width, height, channels = 1;

    // Open input data
    printf("Reading input data\n");
    
    // Read dataset sizes
    size_t train_size = ReadUByteDataset(FLAGS_train_images.c_str(), FLAGS_train_labels.c_str(), nullptr, nullptr, width, height);

    printf("Train Size %ld\n", train_size);

    size_t test_size = ReadUByteDataset(FLAGS_test_images.c_str(), FLAGS_test_labels.c_str(), nullptr, nullptr, width, height);
    if (train_size == 0)
        return 1;
    
    std::vector<uint8_t> train_images(train_size * width * height * channels), train_labels(train_size);
    std::vector<uint8_t> test_images(test_size * width * height * channels), test_labels(test_size);

    // Read data from datasets
    if (ReadUByteDataset(FLAGS_train_images.c_str(), FLAGS_train_labels.c_str(), &train_images[0], &train_labels[0], width, height) != train_size)
        return 2;
    if (ReadUByteDataset(FLAGS_test_images.c_str(), FLAGS_test_labels.c_str(), &test_images[0], &test_labels[0], width, height) != test_size)
        return 3;

    printf("Done. Training dataset size: %d, Test dataset size: %d\n", (int)train_size, (int)test_size);
    printf("Batch size: %lld, iterations: %d\n", FLAGS_batch_size, FLAGS_iterations);

    // This code snippet saves a random image and its label
    /*
    std::random_device rd_image;
    int random_image = rd_image() % train_size;
    std::stringstream ss; ss << "image-" << (int)train_labels[random_image] << ".pgm";
    SavePGMFile(&train_images[0] + random_image * width*height*channels, width, height, ss.str().c_str());
    */

    // Choose GPU
    int no_gpu = NGPUs;
    int num_gpus;
    /*checkCudaErrors(cudaGetDeviceCount(&num_gpus));
    if (FLAGS_gpu < 0 || FLAGS_gpu >= num_gpus)
    {
        printf("ERROR: Invalid GPU ID %d (There are %d GPUs on this machine)\n",
               FLAGS_gpu, num_gpus);
        return 4;
    }*/

    // Create the LeNet network architecture
    ConvBiasLayer conv1((int)channels, 20, 5, (int)width, (int)height);
    MaxPoolLayer pool1(2, 2);
    ConvBiasLayer conv2(conv1.out_channels, 50, 5, conv1.out_width / pool1.stride, conv1.out_height / pool1.stride);
    MaxPoolLayer pool2(2, 2);
    FullyConnectedLayer fc1((conv2.out_channels*conv2.out_width*conv2.out_height) / (pool2.stride * pool2.stride), 
                            500);
    FullyConnectedLayer fc2(fc1.outputs, 10);

    // Initialize CUDNN/CUBLAS training context. #saiful (creating context for all the GPUs)
 //   if(no_gpu == 1) 
    	TrainingContext context(0, FLAGS_batch_size, conv1, pool1, conv2, pool2, fc1, fc2);
   // else if (no_gpu == 2){
    	TrainingContext context_0(dev_0, FLAGS_batch_size/2, conv1, pool1, conv2, pool2, fc1, fc2);
    
    	TrainingContext context_1(dev_1, FLAGS_batch_size/2, conv1, pool1, conv2, pool2, fc1, fc2);
   // }


    // Determine initial network structure
    bool bRet = true;
    if (FLAGS_pretrained)
    {
      bRet = conv1.FromFile("conv1");
      bRet &= conv2.FromFile("conv2");
      bRet &= fc1.FromFile("ip1");
      bRet &= fc2.FromFile("ip2");
    }
    if (!bRet || !FLAGS_pretrained)
    {
        // Create random network
        std::random_device rd;
        std::mt19937 gen(FLAGS_random_seed < 0 ? rd() : static_cast<unsigned int>(FLAGS_random_seed));

        // Xavier weight filling
        float wconv1 = sqrt(3.0f / (conv1.kernel_size * conv1.kernel_size * conv1.in_channels));
        std::uniform_real_distribution<> dconv1(-wconv1, wconv1);
        float wconv2 = sqrt(3.0f / (conv2.kernel_size * conv2.kernel_size * conv2.in_channels));
        std::uniform_real_distribution<> dconv2(-wconv2, wconv2);
        float wfc1 = sqrt(3.0f / (fc1.inputs * fc1.outputs));
        std::uniform_real_distribution<> dfc1(-wfc1, wfc1);
        float wfc2 = sqrt(3.0f / (fc2.inputs * fc2.outputs));
        std::uniform_real_distribution<> dfc2(-wfc2, wfc2);

        // Randomize network
        for (auto&& iter : conv1.pconv)
            iter = static_cast<float>(dconv1(gen));
        for (auto&& iter : conv1.pbias)
            iter = static_cast<float>(dconv1(gen));
        for (auto&& iter : conv2.pconv)
            iter = static_cast<float>(dconv2(gen));
        for (auto&& iter : conv2.pbias)
            iter = static_cast<float>(dconv2(gen));
        for (auto&& iter : fc1.pneurons)
            iter = static_cast<float>(dfc1(gen));
        for (auto&& iter : fc1.pbias)
            iter = static_cast<float>(dfc1(gen));
        for (auto&& iter : fc2.pneurons)
            iter = static_cast<float>(dfc2(gen));
        for (auto&& iter : fc2.pbias)
            iter = static_cast<float>(dfc2(gen));
    }
    
    /////////////////////////////////////////////////////////////////////////////
    // Create GPU data structures    

    // Forward propagation data
    // #saiful: Adding cases for multi-GPU contexts 
    if(no_gpu ==1){
    		float *d_data, *d_labels, *d_conv1, *d_pool1, *d_conv2, *d_pool2, *d_fc1, *d_fc1relu, *d_fc2, *d_fc2smax;
		cudaSetDevice(0);
    //                         Buffer    | Element       | N                   | C                  | H                                 | W
    //-----------------------------------------------------------------------------------------------------------------------------------------
   		 checkCudaErrors(cudaMalloc(&d_data,    sizeof(float) * context.m_batchSize * channels           * height                            * width));
    		checkCudaErrors(cudaMalloc(&d_labels,  sizeof(float) * context.m_batchSize * 1                  * 1                                 * 1));
    		checkCudaErrors(cudaMalloc(&d_conv1,   sizeof(float) * context.m_batchSize * conv1.out_channels * conv1.out_height                  * conv1.out_width));
    		checkCudaErrors(cudaMalloc(&d_pool1,   sizeof(float) * context.m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride)));
    		checkCudaErrors(cudaMalloc(&d_conv2,   sizeof(float) * context.m_batchSize * conv2.out_channels * conv2.out_height                  * conv2.out_width));
    		checkCudaErrors(cudaMalloc(&d_pool2,   sizeof(float) * context.m_batchSize * conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride)));
    		checkCudaErrors(cudaMalloc(&d_fc1,     sizeof(float) * context.m_batchSize * fc1.outputs));    
    		checkCudaErrors(cudaMalloc(&d_fc1relu, sizeof(float) * context.m_batchSize * fc1.outputs));
    		checkCudaErrors(cudaMalloc(&d_fc2,     sizeof(float) * context.m_batchSize * fc2.outputs));
    		checkCudaErrors(cudaMalloc(&d_fc2smax, sizeof(float) * context.m_batchSize * fc2.outputs));    

  // Network parameters
    		float *d_pconv1, *d_pconv1bias, *d_pconv2, *d_pconv2bias;
    		float *d_pfc1, *d_pfc1bias, *d_pfc2, *d_pfc2bias;
    
    		checkCudaErrors(cudaMalloc(&d_pconv1,     sizeof(float) * conv1.pconv.size()));
    		checkCudaErrors(cudaMalloc(&d_pconv1bias, sizeof(float) * conv1.pbias.size()));
    		checkCudaErrors(cudaMalloc(&d_pconv2,     sizeof(float) * conv2.pconv.size()));
    		checkCudaErrors(cudaMalloc(&d_pconv2bias, sizeof(float) * conv2.pbias.size()));
    		checkCudaErrors(cudaMalloc(&d_pfc1,       sizeof(float) * fc1.pneurons.size()));
    		checkCudaErrors(cudaMalloc(&d_pfc1bias,   sizeof(float) * fc1.pbias.size()));
    		checkCudaErrors(cudaMalloc(&d_pfc2,       sizeof(float) * fc2.pneurons.size()));
    		checkCudaErrors(cudaMalloc(&d_pfc2bias,   sizeof(float) * fc2.pbias.size()));    
    
    // Network parameter gradients
    		float *d_gconv1, *d_gconv1bias, *d_gconv2, *d_gconv2bias;
    float *d_gfc1, *d_gfc1bias, *d_gfc2, *d_gfc2bias;
    
    		checkCudaErrors(cudaMalloc(&d_gconv1,     sizeof(float) * conv1.pconv.size()));
    		checkCudaErrors(cudaMalloc(&d_gconv1bias, sizeof(float) * conv1.pbias.size()));
    		checkCudaErrors(cudaMalloc(&d_gconv2,     sizeof(float) * conv2.pconv.size()));
    		checkCudaErrors(cudaMalloc(&d_gconv2bias, sizeof(float) * conv2.pbias.size()));
    		checkCudaErrors(cudaMalloc(&d_gfc1,       sizeof(float) * fc1.pneurons.size()));
    		checkCudaErrors(cudaMalloc(&d_gfc1bias,   sizeof(float) * fc1.pbias.size()));    
    		checkCudaErrors(cudaMalloc(&d_gfc2,       sizeof(float) * fc2.pneurons.size()));
    		checkCudaErrors(cudaMalloc(&d_gfc2bias,   sizeof(float) * fc2.pbias.size()));
    
    // Differentials w.r.t. data
    		float *d_dpool1, *d_dpool2, *d_dconv2, *d_dfc1, *d_dfc1relu, *d_dfc2, *d_dfc2smax, *d_dlossdata;
    //                         Buffer     | Element       | N                   | C                  | H                                 | W
    //-----------------------------------------------------------------------------------------------------------------------------------------
    		checkCudaErrors(cudaMalloc(&d_dpool1,   sizeof(float) * context.m_batchSize * conv1.out_channels * conv1.out_height                  * conv1.out_width));
    		checkCudaErrors(cudaMalloc(&d_dpool2,   sizeof(float) * context.m_batchSize * conv2.out_channels * conv2.out_height                  * conv2.out_width));
    		checkCudaErrors(cudaMalloc(&d_dconv2,   sizeof(float) * context.m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride)));
    		checkCudaErrors(cudaMalloc(&d_dfc1,     sizeof(float) * context.m_batchSize * fc1.inputs));
    		checkCudaErrors(cudaMalloc(&d_dfc1relu, sizeof(float) * context.m_batchSize * fc1.outputs));
    		checkCudaErrors(cudaMalloc(&d_dfc2,     sizeof(float) * context.m_batchSize * fc2.inputs));
    		checkCudaErrors(cudaMalloc(&d_dfc2smax, sizeof(float) * context.m_batchSize * fc2.outputs));
    		checkCudaErrors(cudaMalloc(&d_dlossdata,sizeof(float) * context.m_batchSize * fc2.outputs));
    
    // Temporary buffers and workspaces
    		float *d_onevec;
    		void *d_cudnn_workspace = nullptr;    
    		checkCudaErrors(cudaMalloc(&d_onevec, sizeof(float)* context.m_batchSize));
    		if (context.m_workspaceSize > 0)
        		checkCudaErrors(cudaMalloc(&d_cudnn_workspace, context.m_workspaceSize));    

    /////////////////////////////////////////////////////////////////////////////

    // Copy initial network to device
   		checkCudaErrors(cudaMemcpyAsync(d_pconv1, &conv1.pconv[0],     sizeof(float) * conv1.pconv.size(),  cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpyAsync(d_pconv1bias, &conv1.pbias[0], sizeof(float) * conv1.pbias.size(),  cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpyAsync(d_pconv2, &conv2.pconv[0],     sizeof(float) * conv2.pconv.size(),  cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpyAsync(d_pconv2bias, &conv2.pbias[0], sizeof(float) * conv2.pbias.size(),  cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpyAsync(d_pfc1, &fc1.pneurons[0],      sizeof(float) * fc1.pneurons.size(), cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpyAsync(d_pfc1bias, &fc1.pbias[0],     sizeof(float) * fc1.pbias.size(),    cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpyAsync(d_pfc2, &fc2.pneurons[0],      sizeof(float) * fc2.pneurons.size(), cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpyAsync(d_pfc2bias, &fc2.pbias[0],     sizeof(float) * fc2.pbias.size(),    cudaMemcpyHostToDevice));
    
    // Fill one-vector with ones
    		FillOnes<<<RoundUp(context.m_batchSize, BW), BW>>>(d_onevec, context.m_batchSize);

    		printf("Preparing dataset\n");
    
    // Normalize training set to be in [0,1]
    		std::vector<float> train_images_float(train_images.size()), train_labels_float(train_size);
    		for (size_t i = 0; i < train_size * channels * width * height; ++i)
        		train_images_float[i] = (float)train_images[i] / 255.0f;
    
    		for (size_t i = 0; i < train_size; ++i)
        		train_labels_float[i] = (float)train_labels[i];

    		printf("Training...\n");

    // Use SGD to train the network
    		checkCudaErrors(cudaDeviceSynchronize());
    		auto t1 = std::chrono::high_resolution_clock::now();
    		for (int iter = 0; iter < FLAGS_iterations; ++iter)
    		{
        // Train
        		int imageid = iter % (train_size / context.m_batchSize);
			printf("Image id = %d \n", imageid);
        // Prepare current batch on device
        		checkCudaErrors(cudaMemcpyAsync(d_data, &train_images_float[imageid * context.m_batchSize * width*height*channels],
                                        sizeof(float) * context.m_batchSize * channels * width * height, cudaMemcpyHostToDevice));
        		checkCudaErrors(cudaMemcpyAsync(d_labels, &train_labels_float[imageid * context.m_batchSize],
                                        sizeof(float) * context.m_batchSize, cudaMemcpyHostToDevice));
        
        // Forward propagation
        		context.ForwardPropagation(d_data, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax, 
                                   d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                                   d_cudnn_workspace, d_onevec);

        // Backward propagation
        		context.Backpropagation(conv1, pool1, conv2, pool2,
                                d_data, d_labels, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax, d_dlossdata,
                                d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                                d_gconv1, d_gconv1bias, d_dpool1, d_gconv2, d_gconv2bias, d_dconv2, d_dpool2, d_gfc1, d_gfc1bias, 
                                d_dfc1, d_dfc1relu, d_gfc2, d_gfc2bias, d_dfc2, d_cudnn_workspace, d_onevec);

        // Compute learning rate
        		float learningRate = static_cast<float>(FLAGS_learning_rate * pow((1.0 + FLAGS_lr_gamma * iter), (-FLAGS_lr_power)));

        // Update weights
        		context.UpdateWeights(learningRate, conv1, conv2,
                              d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                              d_gconv1, d_gconv1bias, d_gconv2, d_gconv2bias, d_gfc1, d_gfc1bias, d_gfc2, d_gfc2bias);
    		}
    		checkCudaErrors(cudaDeviceSynchronize());
    		auto t2 = std::chrono::high_resolution_clock::now();

    		printf("Iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / FLAGS_iterations);
    
    		if (FLAGS_save_data)
    		{
        // Copy trained weights from GPU to CPU
        		checkCudaErrors(cudaMemcpy(&conv1.pconv[0], d_pconv1, sizeof(float) * conv1.pconv.size(), cudaMemcpyDeviceToHost));
        		checkCudaErrors(cudaMemcpy(&conv1.pbias[0], d_pconv1bias, sizeof(float) * conv1.pbias.size(), cudaMemcpyDeviceToHost));
        		checkCudaErrors(cudaMemcpy(&conv2.pconv[0], d_pconv2, sizeof(float) * conv2.pconv.size(), cudaMemcpyDeviceToHost));
        		checkCudaErrors(cudaMemcpy(&conv2.pbias[0], d_pconv2bias, sizeof(float) * conv2.pbias.size(), cudaMemcpyDeviceToHost));
        		checkCudaErrors(cudaMemcpy(&fc1.pneurons[0], d_pfc1, sizeof(float) * fc1.pneurons.size(), cudaMemcpyDeviceToHost));
        		checkCudaErrors(cudaMemcpy(&fc1.pbias[0], d_pfc1bias, sizeof(float) * fc1.pbias.size(), cudaMemcpyDeviceToHost));
        		checkCudaErrors(cudaMemcpy(&fc2.pneurons[0], d_pfc2, sizeof(float) * fc2.pneurons.size(), cudaMemcpyDeviceToHost));
        		checkCudaErrors(cudaMemcpy(&fc2.pbias[0], d_pfc2bias, sizeof(float) * fc2.pbias.size(), cudaMemcpyDeviceToHost));
      
        // Now save data
        		printf("Saving data to file\n");
        		conv1.ToFile("conv1");
        		conv2.ToFile("conv2");
        		fc1.ToFile("ip1");
        		fc2.ToFile("ip2");
    		}	
    

    		float classification_error = 1.0f;

    		int classifications = FLAGS_classify;
    		if (classifications < 0)
        		classifications = (int)test_size;
    
    // Test the resulting neural network's classification
    		if (classifications > 0)
    		{
        // Initialize a TrainingContext structure for testing (different batch size)
        		TrainingContext test_context(FLAGS_gpu, 1, conv1, pool1, conv2, pool2, fc1, fc2);

        // Ensure correct workspaceSize is allocated for testing
        		if (context.m_workspaceSize < test_context.m_workspaceSize)
        		{
            		checkCudaErrors(cudaFree(d_cudnn_workspace));
            		checkCudaErrors(cudaMalloc(&d_cudnn_workspace, test_context.m_workspaceSize));
        		}

        		int num_errors = 0;
        		for (int i = 0; i < classifications; ++i)
        		{
            			std::vector<float> data(width * height);
            // Normalize image to be in [0,1]
            			for (int j = 0; j < width * height; ++j)
                			data[j] = (float)test_images[i * width*height*channels + j] / 255.0f;

            			checkCudaErrors(cudaMemcpyAsync(d_data, &data[0], sizeof(float) * width * height, cudaMemcpyHostToDevice));
            
            // Forward propagate test image
            			test_context.ForwardPropagation(d_data, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax,
                                            d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias,
                                            d_pfc2, d_pfc2bias, d_cudnn_workspace, d_onevec);

            // Perform classification
            			std::vector<float> class_vec(10);

            // Copy back result
            			checkCudaErrors(cudaMemcpy(&class_vec[0], d_fc2smax, sizeof(float) * 10, cudaMemcpyDeviceToHost));

            // Determine classification according to maximal response
            			int chosen = 0;
            			for (int id = 1; id < 10; ++id)
            			{	
                			if (class_vec[chosen] < class_vec[id]) chosen = id;
            			}

            		if (chosen != test_labels[i])
                		++num_errors;
        		}
        		classification_error = (float)num_errors / (float)classifications;

        		printf("Classification result: %.2f%% error (used %d images)\n", classification_error * 100.0f, (int)classifications);
    		}
        
    // Free data structures
    		checkCudaErrors(cudaFree(d_data));
    		checkCudaErrors(cudaFree(d_conv1));
    		checkCudaErrors(cudaFree(d_pool1));
    		checkCudaErrors(cudaFree(d_conv2));
    		checkCudaErrors(cudaFree(d_pool2));
    		checkCudaErrors(cudaFree(d_fc1));
    		checkCudaErrors(cudaFree(d_fc2));
    		checkCudaErrors(cudaFree(d_pconv1));
    		checkCudaErrors(cudaFree(d_pconv1bias));
    		checkCudaErrors(cudaFree(d_pconv2));
    		checkCudaErrors(cudaFree(d_pconv2bias));
    		checkCudaErrors(cudaFree(d_pfc1));
    		checkCudaErrors(cudaFree(d_pfc1bias));
    		checkCudaErrors(cudaFree(d_pfc2));
    		checkCudaErrors(cudaFree(d_pfc2bias));
    		checkCudaErrors(cudaFree(d_gconv1));
    		checkCudaErrors(cudaFree(d_gconv1bias));
    		checkCudaErrors(cudaFree(d_gconv2));
    		checkCudaErrors(cudaFree(d_gconv2bias));
    		checkCudaErrors(cudaFree(d_gfc1));
    		checkCudaErrors(cudaFree(d_gfc1bias));
    		checkCudaErrors(cudaFree(d_dfc1));
    		checkCudaErrors(cudaFree(d_gfc2));
    		checkCudaErrors(cudaFree(d_gfc2bias));
    		checkCudaErrors(cudaFree(d_dfc2));
    		checkCudaErrors(cudaFree(d_dpool1));
    		checkCudaErrors(cudaFree(d_dconv2));
    		checkCudaErrors(cudaFree(d_dpool2));    
    		checkCudaErrors(cudaFree(d_labels));
    		checkCudaErrors(cudaFree(d_dlossdata));
    		checkCudaErrors(cudaFree(d_onevec));
    		if (d_cudnn_workspace != nullptr)
        		checkCudaErrors(cudaFree(d_cudnn_workspace));

    		return 0;
		
    }
    else if (no_gpu == 2){
	
    		
		
		float *d_data_0, *d_labels_0, *d_conv1_0, *d_pool1_0, *d_conv2_0, *d_pool2_0, *d_fc1_0, *d_fc1relu_0, *d_fc2_0, *d_fc2smax_0;
		float *d_data_1, *d_labels_1, *d_conv1_1, *d_pool1_1, *d_conv2_1, *d_pool2_1, *d_fc1_1, *d_fc1relu_1, *d_fc2_1, *d_fc2smax_1;
    //                         Buffer    | Element       | N                   | C                  | H                                 | W
    //-----------------------------------------------------------------------------------------------------------------------------------------
   		checkCudaErrors(cudaSetDevice(dev_0));

		checkCudaErrors(cudaMalloc(&d_data_0,    sizeof(float) * context_0.m_batchSize * channels           * height                            * width));
    		checkCudaErrors(cudaMalloc(&d_labels_0,  sizeof(float) * context_0.m_batchSize * 1                  * 1                                 * 1));
    		checkCudaErrors(cudaMalloc(&d_conv1_0,   sizeof(float) * context_0.m_batchSize * conv1.out_channels * conv1.out_height                  * conv1.out_width));
    		checkCudaErrors(cudaMalloc(&d_pool1_0,   sizeof(float) * context_0.m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride)));
    		checkCudaErrors(cudaMalloc(&d_conv2_0,   sizeof(float) * context_0.m_batchSize * conv2.out_channels * conv2.out_height                  * conv2.out_width));
    		checkCudaErrors(cudaMalloc(&d_pool2_0,   sizeof(float) * context_0.m_batchSize * conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride)));
    		checkCudaErrors(cudaMalloc(&d_fc1_0,     sizeof(float) * context_0.m_batchSize * fc1.outputs));    
    		checkCudaErrors(cudaMalloc(&d_fc1relu_0, sizeof(float) * context_0.m_batchSize * fc1.outputs));
    		checkCudaErrors(cudaMalloc(&d_fc2_0,     sizeof(float) * context_0.m_batchSize * fc2.outputs));
    		checkCudaErrors(cudaMalloc(&d_fc2smax_0, sizeof(float) * context_0.m_batchSize * fc2.outputs));    

   		checkCudaErrors(cudaSetDevice(dev_1));

		checkCudaErrors(cudaMalloc(&d_data_1,    sizeof(float) * context_1.m_batchSize * channels           * height                            * width));
    		checkCudaErrors(cudaMalloc(&d_labels_1,  sizeof(float) * context_1.m_batchSize * 1                  * 1                                 * 1));
    		checkCudaErrors(cudaMalloc(&d_conv1_1,   sizeof(float) * context_1.m_batchSize * conv1.out_channels * conv1.out_height                  * conv1.out_width));
    		checkCudaErrors(cudaMalloc(&d_pool1_1,   sizeof(float) * context_1.m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride)));
    		checkCudaErrors(cudaMalloc(&d_conv2_1,   sizeof(float) * context_1.m_batchSize * conv2.out_channels * conv2.out_height                  * conv2.out_width));
    		checkCudaErrors(cudaMalloc(&d_pool2_1,   sizeof(float) * context_1.m_batchSize * conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride)));
    		checkCudaErrors(cudaMalloc(&d_fc1_1,     sizeof(float) * context_1.m_batchSize * fc1.outputs));    
    		checkCudaErrors(cudaMalloc(&d_fc1relu_1, sizeof(float) * context_1.m_batchSize * fc1.outputs));
    		checkCudaErrors(cudaMalloc(&d_fc2_1,     sizeof(float) * context_1.m_batchSize * fc2.outputs));
    		checkCudaErrors(cudaMalloc(&d_fc2smax_1, sizeof(float) * context_1.m_batchSize * fc2.outputs));    
  // Network parameters
    		float *d_pconv1_1, *d_pconv1bias_1, *d_pconv2_1, *d_pconv2bias_1;
    		float *d_pfc1_1, *d_pfc1bias_1, *d_pfc2_1, *d_pfc2bias_1;
    
   		checkCudaErrors(cudaSetDevice(dev_1));
    		checkCudaErrors(cudaMalloc(&d_pconv1_1,     sizeof(float) * conv1.pconv.size()));
    		checkCudaErrors(cudaMalloc(&d_pconv1bias_1, sizeof(float) * conv1.pbias.size()));
    		checkCudaErrors(cudaMalloc(&d_pconv2_1,     sizeof(float) * conv2.pconv.size()));
    		checkCudaErrors(cudaMalloc(&d_pconv2bias_1, sizeof(float) * conv2.pbias.size()));
    		checkCudaErrors(cudaMalloc(&d_pfc1_1,       sizeof(float) * fc1.pneurons.size()));
    		checkCudaErrors(cudaMalloc(&d_pfc1bias_1,   sizeof(float) * fc1.pbias.size()));
    		checkCudaErrors(cudaMalloc(&d_pfc2_1,       sizeof(float) * fc2.pneurons.size()));
    		checkCudaErrors(cudaMalloc(&d_pfc2bias_1,   sizeof(float) * fc2.pbias.size()));    
    		
   		checkCudaErrors(cudaSetDevice(dev_0));


		float *d_pconv1_0, *d_pconv1bias_0, *d_pconv2_0, *d_pconv2bias_0;
    		float *d_pfc1_0, *d_pfc1bias_0, *d_pfc2_0, *d_pfc2bias_0;
    
    		checkCudaErrors(cudaMalloc(&d_pconv1_0,     sizeof(float) * conv1.pconv.size()));
    		checkCudaErrors(cudaMalloc(&d_pconv1bias_0, sizeof(float) * conv1.pbias.size()));
    		checkCudaErrors(cudaMalloc(&d_pconv2_0,     sizeof(float) * conv2.pconv.size()));
    		checkCudaErrors(cudaMalloc(&d_pconv2bias_0, sizeof(float) * conv2.pbias.size()));
    		checkCudaErrors(cudaMalloc(&d_pfc1_0,       sizeof(float) * fc1.pneurons.size()));
    		checkCudaErrors(cudaMalloc(&d_pfc1bias_0,   sizeof(float) * fc1.pbias.size()));
    		checkCudaErrors(cudaMalloc(&d_pfc2_0,       sizeof(float) * fc2.pneurons.size()));
    		checkCudaErrors(cudaMalloc(&d_pfc2bias_0,   sizeof(float) * fc2.pbias.size()));    
    		checkCudaErrors(cudaMalloc(&d_pconv1_0,     sizeof(float) * conv1.pconv.size()));
    		checkCudaErrors(cudaMalloc(&d_pconv1bias_0, sizeof(float) * conv1.pbias.size()));
    		checkCudaErrors(cudaMalloc(&d_pconv2_0,     sizeof(float) * conv2.pconv.size()));
    		checkCudaErrors(cudaMalloc(&d_pconv2bias_0, sizeof(float) * conv2.pbias.size()));
    		checkCudaErrors(cudaMalloc(&d_pfc1_0,       sizeof(float) * fc1.pneurons.size()));
    		checkCudaErrors(cudaMalloc(&d_pfc1bias_0,   sizeof(float) * fc1.pbias.size()));
    		checkCudaErrors(cudaMalloc(&d_pfc2_0,       sizeof(float) * fc2.pneurons.size()));
    		checkCudaErrors(cudaMalloc(&d_pfc2bias_0,   sizeof(float) * fc2.pbias.size()));    
    
    // Network parameter gradients
    		float *d_gconv1_0, *d_gconv1bias_0, *d_gconv2_0, *d_gconv2bias_0;
    		float *d_gfc1_0, *d_gfc1bias_0, *d_gfc2_0, *d_gfc2bias_0;
    
    		float *d_gconv1_1, *d_gconv1bias_1, *d_gconv2_1, *d_gconv2bias_1;
    		float *d_gfc1_1, *d_gfc1bias_1, *d_gfc2_1, *d_gfc2bias_1;
    		
   		checkCudaErrors(cudaSetDevice(dev_1));

		checkCudaErrors(cudaMalloc(&d_gconv1_1,     sizeof(float) * conv1.pconv.size()));
    		checkCudaErrors(cudaMalloc(&d_gconv1bias_1, sizeof(float) * conv1.pbias.size()));
    		checkCudaErrors(cudaMalloc(&d_gconv2_1,     sizeof(float) * conv2.pconv.size()));
    		checkCudaErrors(cudaMalloc(&d_gconv2bias_1, sizeof(float) * conv2.pbias.size()));
    		checkCudaErrors(cudaMalloc(&d_gfc1_1,       sizeof(float) * fc1.pneurons.size()));
    		checkCudaErrors(cudaMalloc(&d_gfc1bias_1,   sizeof(float) * fc1.pbias.size()));    
    		checkCudaErrors(cudaMalloc(&d_gfc2_1,       sizeof(float) * fc2.pneurons.size()));
    		checkCudaErrors(cudaMalloc(&d_gfc2bias_1,   sizeof(float) * fc2.pbias.size()));
    		
		checkCudaErrors(cudaSetDevice(dev_0));

    		
		checkCudaErrors(cudaMalloc(&d_gconv1_0,     sizeof(float) * conv1.pconv.size()));
    		checkCudaErrors(cudaMalloc(&d_gconv1bias_0, sizeof(float) * conv1.pbias.size()));
    		checkCudaErrors(cudaMalloc(&d_gconv2_0,     sizeof(float) * conv2.pconv.size()));
    		checkCudaErrors(cudaMalloc(&d_gconv2bias_0, sizeof(float) * conv2.pbias.size()));
    		checkCudaErrors(cudaMalloc(&d_gfc1_0,       sizeof(float) * fc1.pneurons.size()));
    		checkCudaErrors(cudaMalloc(&d_gfc1bias_0,   sizeof(float) * fc1.pbias.size()));    
    		checkCudaErrors(cudaMalloc(&d_gfc2_0,       sizeof(float) * fc2.pneurons.size()));
    		checkCudaErrors(cudaMalloc(&d_gfc2bias_0,   sizeof(float) * fc2.pbias.size()));

    // Differentials w.r.t. data
    		float *d_dpool1_0, *d_dpool2_0, *d_dconv2_0, *d_dfc1_0, *d_dfc1relu_0, *d_dfc2_0, *d_dfc2max_0, *d_dlossdata_0;
    		float *d_dpool1_1, *d_dpool2_1, *d_dconv2_1, *d_dfc1_1, *d_dfc1relu_1, *d_dfc2_1, *d_dfc2max_1, *d_dlossdata_1;
    //                         Buffer     | Element       | N                   | C                  | H                                 | W
    //-----------------------------------------------------------------------------------------------------------------------------------------
    		checkCudaErrors(cudaMalloc(&d_dpool1_0,   sizeof(float) * context_0.m_batchSize * conv1.out_channels * conv1.out_height                  * conv1.out_width));
    		checkCudaErrors(cudaMalloc(&d_dpool2_0,   sizeof(float) * context_0.m_batchSize * conv2.out_channels * conv2.out_height                  * conv2.out_width));
    		checkCudaErrors(cudaMalloc(&d_dconv2_0,   sizeof(float) * context_0.m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride)));
    		checkCudaErrors(cudaMalloc(&d_dfc1_0,     sizeof(float) * context_0.m_batchSize * fc1.inputs));
    		checkCudaErrors(cudaMalloc(&d_dfc1relu_0, sizeof(float) * context_0.m_batchSize * fc1.outputs));
    		checkCudaErrors(cudaMalloc(&d_dfc2_0,     sizeof(float) * context_0.m_batchSize * fc2.inputs));
    		checkCudaErrors(cudaMalloc(&d_dfc2max_0, sizeof(float) * context_0.m_batchSize * fc2.outputs));
    		checkCudaErrors(cudaMalloc(&d_dlossdata_0,sizeof(float) * context_0.m_batchSize * fc2.outputs));
   

		checkCudaErrors(cudaSetDevice(dev_1));

    		checkCudaErrors(cudaMalloc(&d_dpool1_1,   sizeof(float) * context_1.m_batchSize * conv1.out_channels * conv1.out_height                  * conv1.out_width));
    		checkCudaErrors(cudaMalloc(&d_dpool2_1,   sizeof(float) * context_1.m_batchSize * conv2.out_channels * conv2.out_height                  * conv2.out_width));
    		checkCudaErrors(cudaMalloc(&d_dconv2_1,   sizeof(float) * context_1.m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride)));
    		checkCudaErrors(cudaMalloc(&d_dfc1_1,     sizeof(float) * context_1.m_batchSize * fc1.inputs));
    		checkCudaErrors(cudaMalloc(&d_dfc1relu_1, sizeof(float) * context_1.m_batchSize * fc1.outputs));
    		checkCudaErrors(cudaMalloc(&d_dfc2_1,     sizeof(float) * context_1.m_batchSize * fc2.inputs));
    		checkCudaErrors(cudaMalloc(&d_dfc2max_1, sizeof(float) * context_1.m_batchSize * fc2.outputs));
    		checkCudaErrors(cudaMalloc(&d_dlossdata_1,sizeof(float) * context_1.m_batchSize * fc2.outputs));

    // Temporary buffers and workspaces
    		float *d_onevec_0;
    		void *d_cudnn_workspace_0 = nullptr;    
    		float *d_onevec_1;
    		void *d_cudnn_workspace_1 = nullptr;    
    		checkCudaErrors(cudaMalloc(&d_onevec_1, sizeof(float)* context_1.m_batchSize));
    		if (context.m_workspaceSize > 0)
        		checkCudaErrors(cudaMalloc(&d_cudnn_workspace_1, context_1.m_workspaceSize));    

		checkCudaErrors(cudaSetDevice(dev_0));
    		
		checkCudaErrors(cudaMalloc(&d_onevec_0, sizeof(float)* context_0.m_batchSize));
    		if (context.m_workspaceSize > 0)
        		checkCudaErrors(cudaMalloc(&d_cudnn_workspace_0, context_0.m_workspaceSize));    

    /////////////////////////////////////////////////////////////////////////////
// changed till this part
    // Copy initial network to device
   		checkCudaErrors(cudaSetDevice(dev_0));
   		checkCudaErrors(cudaMemcpyAsync(d_pconv1_0, &conv1.pconv[0],     sizeof(float) * conv1.pconv.size(),  cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpyAsync(d_pconv1bias_0, &conv1.pbias[0], sizeof(float) * conv1.pbias.size(),  cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpyAsync(d_pconv2_0, &conv2.pconv[0],     sizeof(float) * conv2.pconv.size(),  cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpyAsync(d_pconv2bias_0, &conv2.pbias[0], sizeof(float) * conv2.pbias.size(),  cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpyAsync(d_pfc1_0, &fc1.pneurons[0],      sizeof(float) * fc1.pneurons.size(), cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpyAsync(d_pfc1bias_0, &fc1.pbias[0],     sizeof(float) * fc1.pbias.size(),    cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpyAsync(d_pfc2_0, &fc2.pneurons[0],      sizeof(float) * fc2.pneurons.size(), cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpyAsync(d_pfc2bias_0, &fc2.pbias[0],     sizeof(float) * fc2.pbias.size(),    cudaMemcpyHostToDevice));
    
		
		checkCudaErrors(cudaSetDevice(dev_1));
   // Copy initial network to device
   		checkCudaErrors(cudaMemcpyAsync(d_pconv1_1, &conv1.pconv[0],     sizeof(float) * conv1.pconv.size(),  cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpyAsync(d_pconv1bias_1, &conv1.pbias[0], sizeof(float) * conv1.pbias.size(),  cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpyAsync(d_pconv2_1, &conv2.pconv[0],     sizeof(float) * conv2.pconv.size(),  cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpyAsync(d_pconv2bias_1, &conv2.pbias[0], sizeof(float) * conv2.pbias.size(),  cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpyAsync(d_pfc1_1, &fc1.pneurons[0],      sizeof(float) * fc1.pneurons.size(), cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpyAsync(d_pfc1bias_1, &fc1.pbias[0],     sizeof(float) * fc1.pbias.size(),    cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpyAsync(d_pfc2_1, &fc2.pneurons[0],      sizeof(float) * fc2.pneurons.size(), cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpyAsync(d_pfc2bias_1, &fc2.pbias[0],     sizeof(float) * fc2.pbias.size(),    cudaMemcpyHostToDevice));
    
    // Fill one-vector with ones
   		checkCudaErrors(cudaSetDevice(dev_1));
    		FillOnes<<<RoundUp(context_1.m_batchSize, BW), BW>>>(d_onevec_1, context_1.m_batchSize);
		
		checkCudaErrors(cudaSetDevice(dev_0));
    		
		FillOnes<<<RoundUp(context_0.m_batchSize, BW), BW>>>(d_onevec_0, context_0.m_batchSize);

    		printf("Preparing dataset\n");
    
    // Normalize training set to be in [0,1]
    		std::vector<float> train_images_float(train_images.size()), train_labels_float(train_size);
    		for (size_t i = 0; i < train_size * channels * width * height; ++i)
        		train_images_float[i] = (float)train_images[i] / 255.0f;
    
    		for (size_t i = 0; i < train_size; ++i)
        		train_labels_float[i] = (float)train_labels[i];

    		printf("Training...\n");

    // Use SGD to train the network
    		checkCudaErrors(cudaDeviceSynchronize());
    		auto t1 = std::chrono::high_resolution_clock::now();
    		for (int iter = 0; iter < FLAGS_iterations; ++iter)
    		{
        // Train
				int imageid_0 = 2*iter % (train_size / context_0.m_batchSize);
				int imageid_1 = imageid_0 + 1;
		//	printf("Image id = %d \n", imageid);
        // Prepare current batch on device
		checkCudaErrors(cudaSetDevice(dev_0));
        		checkCudaErrors(cudaMemcpyAsync(d_data_0, &train_images_float[imageid_0 * (context_0.m_batchSize) * width*height*channels],
                                        sizeof(float) * context_0.m_batchSize * channels * width * height, cudaMemcpyHostToDevice));
        		checkCudaErrors(cudaMemcpyAsync(d_labels_0, &train_labels_float[imageid_0 * context_0.m_batchSize],
                                        sizeof(float) * context_0.m_batchSize, cudaMemcpyHostToDevice));
       
		checkCudaErrors(cudaSetDevice(dev_1));
        		checkCudaErrors(cudaMemcpyAsync(d_data_1, &train_images_float[imageid_1 * (context_1.m_batchSize) * width*height*channels],
                                        sizeof(float) * context_1.m_batchSize * channels * width * height, cudaMemcpyHostToDevice));
        		checkCudaErrors(cudaMemcpyAsync(d_labels_1, &train_labels_float[imageid_1 * context_1.m_batchSize],
                                        sizeof(float) * context_1.m_batchSize, cudaMemcpyHostToDevice));

/*
			float *test =  (float*)malloc( sizeof(float) * context_0.m_batchSize * width *  channels * height);
        		cudaSetDevice(0);
        		checkCudaErrors(cudaMemcpyAsync( test, d_data_0, sizeof(float) * context_0.m_batchSize * width * channels * height, cudaMemcpyDeviceToHost));
			for (int kk = 0; kk< context.m_batchSize * channels * height * width; kk++){
				printf("v=%f, %d\n", test[kk], kk);

			}
*/
        // creating data for parallel threads
			pthread_t *id;
			id = (pthread_t*) malloc(sizeof(pthread_t) * no_gpu);
			thread_data data_array[2] = {
				thread_data(context_0, conv1, pool1, conv2, pool2, fc1, fc2, d_data_0, d_labels_0, d_conv1_0, d_pool1_0, d_conv2_0, d_pool2_0, d_fc1_0, d_fc1relu_0, d_fc2_0, d_fc2smax_0, d_dlossdata_0, d_pconv1_0, d_pconv1bias_0, d_pconv2_0, d_pconv2bias_0, d_pfc1_0, d_pfc1bias_0, d_pfc2_0, d_pfc2bias_0, d_gconv1_0, d_gconv1bias_0, d_dpool1_0, d_gconv2_0, d_gconv2bias_0, d_dconv2_0, d_dpool2_0, d_gfc1_0, d_gfc1bias_0, d_dfc1_0, d_dfc1relu_0, d_gfc2_0, d_gfc2bias_0, d_dfc2_0, d_onevec_0, d_cudnn_workspace_0),
	
				thread_data(context_1, conv1, pool1, conv2, pool2, fc1, fc2, d_data_1, d_labels_1, d_conv1_1, d_pool1_1, d_conv2_1, d_pool2_1, d_fc1_1, d_fc1relu_1, d_fc2_1, d_fc2smax_1, d_dlossdata_1, d_pconv1_1, d_pconv1bias_1, d_pconv2_1, d_pconv2bias_1, d_pfc1_1, d_pfc1bias_1, d_pfc2_1, d_pfc2bias_1, d_gconv1_1, d_gconv1bias_1, d_dpool1_1, d_gconv2_1, d_gconv2bias_1, d_dconv2_1, d_dpool2_1, d_gfc1_1, d_gfc1bias_1, d_dfc1_1, d_dfc1relu_1, d_gfc2_1, d_gfc2bias_1, d_dfc2_1, d_onevec_1, d_cudnn_workspace_1)
,
			};
			/*
			data_array[0].d_data = d_data_0;
			data_array[0].d_labels = d_labels_0;
			data_array[0].d_conv1 = d_conv1_0;
			data_array[0].d_conv2 = d_conv2_0;
			data_array[0].d_pool2 = d_pool2_0;
			data_array[0].d_fc1 = d_fc1_0;
			data_array[0].d_fc1relu = d_fc1relu_0;
			data_array[0].d_fc2 = d_fc2_0;
			data_array[0].d_fc2smax = d_fc2smax_0;
			data_array[0].d_pconv1 = d_pconv1_0;
			data_array[0].d_pconv1bias = d_pconv1bias_0;
			data_array[0].d_pconv2 = d_pconv2_0;
			data_array[0].d_pconv2bias = d_pconv2bias_0;
			data_array[0].d_pfc1 = d_pfc1_0;
			data_array[0].d_pfc1bias = d_pfc1bias_0;
			data_array[0].d_pfc2 = d_pfc2_0;
			data_array[0].d_pfc2bias = d_pfc2bias_0;
			data_array[0].workspace = d_cudnn_workspace_0;
			data_array[0].d_gconv1 = d_gconv1_0;
			data_array[0].d_gconv1bias = d_gconv1bias_0;
			data_array[0].d_dpool1 = d_dpool1_0;
			data_array[0].d_gconv2 = d_gconv2_0;
			data_array[0].d_gconv2bias = d_gconv2bias_0;
			data_array[0].d_dconv2 = d_dconv2_0;
			data_array[0].d_dpool2 = d_dpool2_0;
			data_array[0].d_gfc1 = d_gfc1_0;
			data_array[0].d_gfc1bias = d_gfc1bias_0;
			data_array[0].d_dfc1 = d_dfc1_0;
			data_array[0].d_dfc1relu = d_dfc1relu_0;
			data_array[0].d_gfc2 = d_gfc2_0;
			data_array[0].d_gfc2bias = d_gfc2bias_0;
			data_array[0].d_dfc2 = d_dfc2_0;
			data_array[0].d_onevec = d_onevec_0;
			//data_array[0].conv1 = conv1;
			//data_array[0].pool1 = pool1;
			//data_array[0].conv2 = conv2;
			//data_array[0].pool2 = pool2;
			data_array[0].dev_id = dev_0;
		       	


			data_array[1].d_data = d_data_1;
			data_array[1].d_labels = d_labels_1;
			data_array[1].d_conv1 = d_conv1_1;
			data_array[1].d_conv2 = d_conv2_1;
			data_array[1].d_pool2 = d_pool2_1;
			data_array[1].d_fc1 = d_fc1_1;
			data_array[1].d_fc1relu = d_fc1relu_1;
			data_array[1].d_fc2 = d_fc2_1;
			data_array[1].d_fc2smax = d_fc2smax_1;
			data_array[1].d_pconv1 = d_pconv1_1;
			data_array[1].d_pconv1bias = d_pconv1bias_1;
			data_array[1].d_pconv2 = d_pconv2_1;
			data_array[1].d_pconv2bias = d_pconv2bias_1;
			data_array[1].d_pfc1 = d_pfc1_1;
			data_array[1].d_pfc1bias = d_pfc1bias_1;
			data_array[1].d_pfc2 = d_pfc2_1;
			data_array[1].d_pfc2bias = d_pfc2bias_1;
			data_array[1].workspace = d_cudnn_workspace_1;
			data_array[1].d_gconv1 = d_gconv1_1;
			data_array[1].d_gconv1bias = d_gconv1bias_1;
			data_array[1].d_dpool1 = d_dpool1_1;
			data_array[1].d_gconv2 = d_gconv2_1;
			data_array[1].d_gconv2bias = d_gconv2bias_1;
			data_array[1].d_dconv2 = d_dconv2_1;
			data_array[1].d_dpool2 = d_dpool2_1;
			data_array[1].d_gfc1 = d_gfc1_1;
			data_array[1].d_gfc1bias = d_gfc1bias_1;
			data_array[1].d_dfc1 = d_dfc1_1;
			data_array[1].d_dfc1relu = d_dfc1relu_1;
			data_array[1].d_gfc2 = d_gfc2_1;
			data_array[1].d_gfc2bias = d_gfc2bias_1;
			data_array[1].d_dfc2 = d_dfc2_1;
			data_array[1].d_onevec = d_onevec_1;
		//	data_array[1].conv1 = conv1;
		//	data_array[1].pool1 = pool1;
		//	data_array[1].conv2 = conv2;
		//	data_array[1].pool2 = pool2;
		//	data_array[1].dev_id = dev_1;

			*/
			for (int iii = 0; iii < no_gpu; ++iii){
				if(pthread_create(&id[iii], NULL, training_division, (void*) &data_array[iii])){
					printf("Error creating pthreads\n");
					exit(19);
				}
		
			}


        // Forward propagation
  /*      		context.ForwardPropagation(d_data, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax, 
                                   d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                                   d_cudnn_workspace, d_onevec);

        // Backward propagation
        		context.Backpropagation(conv1, pool1, conv2, pool2,
                                d_data, d_labels, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax, d_dlossdata,
                                d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                                d_gconv1, d_gconv1bias, d_dpool1, d_gconv2, d_gconv2bias, d_dconv2, d_dpool2, d_gfc1, d_gfc1bias, 
                                d_dfc1, d_dfc1relu, d_gfc2, d_gfc2bias, d_dfc2, d_cudnn_workspace, d_onevec);
*/



			for (int iii = 0; iii < no_gpu; iii++) {
				if (pthread_join(id[iii],NULL)){
					exit(19);
				}
			}

/*
			float *test1 =  (float*)malloc( sizeof(float) * context_0.m_batchSize * width *  channels * height);
        		cudaSetDevice(0);
        		checkCudaErrors(cudaMemcpyAsync( test1, d_data_0, sizeof(float) * context_0.m_batchSize * width * channels * height, cudaMemcpyDeviceToHost));
			for (int kk = 0; kk< context.m_batchSize * channels * height * width; kk++){
				printf("v=%f, %d\n", test1[kk], kk);

			}
*/		

			cudaSetDevice(0);
        		cudaDeviceEnablePeerAccess(1, 0);
        		cudaSetDevice(1);
        		cudaDeviceEnablePeerAccess(0, 0);
        // Compute learning rate
        		float learningRate = static_cast<float>(FLAGS_learning_rate * pow((1.0 + FLAGS_lr_gamma * iter), (-FLAGS_lr_power)));







			cudaSetDevice(dev_0);
	
	// Transferring SGDs  from GPU1 to GPU0

    			float *d_gconv1, *d_gconv1bias, *d_gconv2, *d_gconv2bias;
    			float *d_gfc1, *d_gfc1bias, *d_gfc2, *d_gfc2bias;
    
   	 		checkCudaErrors(cudaMalloc(&d_gconv1,     sizeof(float) * conv1.pconv.size()));
    			checkCudaErrors(cudaMalloc(&d_gconv1bias, sizeof(float) * conv1.pbias.size()));
    			checkCudaErrors(cudaMalloc(&d_gconv2,     sizeof(float) * conv2.pconv.size()));
	    		checkCudaErrors(cudaMalloc(&d_gconv2bias, sizeof(float) * conv2.pbias.size()));
    			checkCudaErrors(cudaMalloc(&d_gfc1,       sizeof(float) * fc1.pneurons.size()));
    			checkCudaErrors(cudaMalloc(&d_gfc1bias,   sizeof(float) * fc1.pbias.size()));
    			checkCudaErrors(cudaMalloc(&d_gfc2,       sizeof(float) * fc2.pneurons.size()));
    			checkCudaErrors(cudaMalloc(&d_gfc2bias,   sizeof(float) * fc2.pbias.size()));    

        		checkCudaErrors(cudaMemcpy( d_gconv1, d_gconv1_1,sizeof(float) * conv1.pconv.size(), cudaMemcpyDefault));
        		checkCudaErrors(cudaMemcpy( d_gconv1bias, d_gconv1bias_1, sizeof(float) * conv1.pbias.size(), cudaMemcpyDefault));
        		checkCudaErrors(cudaMemcpy( d_gconv2, d_gconv2_1, sizeof(float) * conv2.pconv.size(), cudaMemcpyDefault));
        		checkCudaErrors(cudaMemcpy(d_gconv2bias, d_gconv2bias_1, sizeof(float) * conv2.pbias.size(), cudaMemcpyDefault));
        		checkCudaErrors(cudaMemcpy(d_gfc1, d_gfc1_1, sizeof(float) * fc1.pneurons.size(), cudaMemcpyDefault));
        		checkCudaErrors(cudaMemcpy(d_gfc1bias, d_gfc1bias_1, sizeof(float) * fc1.pbias.size(), cudaMemcpyDefault));
        		checkCudaErrors(cudaMemcpy(d_gfc2, d_gfc2_1, sizeof(float) * fc2.pneurons.size(), cudaMemcpyDefault));
        		checkCudaErrors(cudaMemcpy(d_gfc2bias, d_gfc2bias_1, sizeof(float) * fc2.pbias.size(), cudaMemcpyDefault));
	
	// Aggregating the SGDs in GPU0
			sync_grad(conv1.pconv.size(), d_gconv1, d_gconv1_0);
			sync_grad(conv1.pbias.size(), d_gconv1bias, d_gconv1bias_0);
			sync_grad(conv2.pconv.size(), d_gconv2, d_gconv2_0);
			sync_grad(conv2.pbias.size(), d_gconv2bias, d_gconv2bias_0);
			sync_grad(fc1.pneurons.size(), d_gfc1, d_gfc1_0);
			sync_grad(fc1.pbias.size(), d_gfc1bias, d_gfc1bias_0);
			sync_grad(fc2.pneurons.size(), d_gfc2, d_gfc2_0);
			sync_grad(fc2.pbias.size(), d_gfc2bias, d_gfc2bias_0);

        // Update weights
        		context_0.UpdateWeights(learningRate, conv1, conv2,
                              d_pconv1_0, d_pconv1bias_0, d_pconv2_0, d_pconv2bias_0, d_pfc1_0, d_pfc1bias_0, d_pfc2_0, d_pfc2bias_0,
                              d_gconv1_0, d_gconv1bias_0, d_gconv2_0, d_gconv2bias_0, d_gfc1_0, d_gfc1bias_0, d_gfc2_0, d_gfc2bias_0);



		// Transferring the updated weights to GPU1 from GPU0

        		checkCudaErrors(cudaMemcpy( d_pconv1_1, d_pconv1_0, sizeof(float) * conv1.pconv.size(), cudaMemcpyDefault));
        		checkCudaErrors(cudaMemcpy( d_pconv1bias_1, d_pconv1bias_0, sizeof(float) * conv1.pbias.size(), cudaMemcpyDefault));
        		checkCudaErrors(cudaMemcpy( d_pconv2_1, d_pconv2_0, sizeof(float) * conv2.pconv.size(), cudaMemcpyDefault));
        		checkCudaErrors(cudaMemcpy( d_pconv2bias_1, d_pconv2bias_0, sizeof(float) * conv2.pbias.size(), cudaMemcpyDefault));
        		checkCudaErrors(cudaMemcpy(d_pfc1_1, d_pfc1_0, sizeof(float) * fc1.pneurons.size(), cudaMemcpyDefault));
        		checkCudaErrors(cudaMemcpy( d_pfc1bias_1, d_pfc1bias_0, sizeof(float) * fc1.pbias.size(), cudaMemcpyDefault));
        		checkCudaErrors(cudaMemcpy(d_pfc2_1, d_pfc2_0, sizeof(float) * fc2.pneurons.size(), cudaMemcpyDefault));
        		checkCudaErrors(cudaMemcpy( d_pfc2bias_1, d_pfc2bias_0, sizeof(float) * fc2.pbias.size(), cudaMemcpyDefault));
    		}
    		checkCudaErrors(cudaDeviceSynchronize());
    		auto t2 = std::chrono::high_resolution_clock::now();

    		printf("Iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / FLAGS_iterations);
    
    		if (FLAGS_save_data)
    		{
        // Copy trained weights from GPU to CPU
        		checkCudaErrors(cudaMemcpy(&conv1.pconv[0], d_pconv1_0, sizeof(float) * conv1.pconv.size(), cudaMemcpyDeviceToHost));
        		checkCudaErrors(cudaMemcpy(&conv1.pbias[0], d_pconv1bias_0, sizeof(float) * conv1.pbias.size(), cudaMemcpyDeviceToHost));
        		checkCudaErrors(cudaMemcpy(&conv2.pconv[0], d_pconv2_0, sizeof(float) * conv2.pconv.size(), cudaMemcpyDeviceToHost));
        		checkCudaErrors(cudaMemcpy(&conv2.pbias[0], d_pconv2bias_0, sizeof(float) * conv2.pbias.size(), cudaMemcpyDeviceToHost));
        		checkCudaErrors(cudaMemcpy(&fc1.pneurons[0], d_pfc1_0, sizeof(float) * fc1.pneurons.size(), cudaMemcpyDeviceToHost));
        		checkCudaErrors(cudaMemcpy(&fc1.pbias[0], d_pfc1bias_0, sizeof(float) * fc1.pbias.size(), cudaMemcpyDeviceToHost));
        		checkCudaErrors(cudaMemcpy(&fc2.pneurons[0], d_pfc2_0, sizeof(float) * fc2.pneurons.size(), cudaMemcpyDeviceToHost));
        		checkCudaErrors(cudaMemcpy(&fc2.pbias[0], d_pfc2bias_0, sizeof(float) * fc2.pbias.size(), cudaMemcpyDeviceToHost));
      
        // Now save data
        		printf("Saving data to file\n");
        		conv1.ToFile("conv1");
        		conv2.ToFile("conv2");
        		fc1.ToFile("ip1");
        		fc2.ToFile("ip2");
    		}	
    

    		float classification_error = 1.0f;

    		int classifications = FLAGS_classify;
    		if (classifications < 0)
        		classifications = (int)test_size;
    
    // Test the resulting neural network's classification
    		if (classifications > 0)
    		{
        // Initialize a TrainingContext structure for testing (different batch size)
        		TrainingContext test_context(dev_0, 1, conv1, pool1, conv2, pool2, fc1, fc2);

        // Ensure correct workspaceSize is allocated for testing
        		if (context_0.m_workspaceSize < test_context.m_workspaceSize)
        		{
            		checkCudaErrors(cudaFree(d_cudnn_workspace_0));
            		checkCudaErrors(cudaMalloc(&d_cudnn_workspace_0, test_context.m_workspaceSize));
        		}

        		int num_errors = 0;
        		for (int i = 0; i < classifications; ++i)
        		{
            			std::vector<float> data(width * height);
            // Normalize image to be in [0,1]
            			for (int j = 0; j < width * height; ++j)
                			data[j] = (float)test_images[i * width*height*channels + j] / 255.0f;

            			checkCudaErrors(cudaMemcpyAsync(d_data_0, &data[0], sizeof(float) * width * height, cudaMemcpyHostToDevice));
            
            // Forward propagate test image
            			test_context.ForwardPropagation(d_data_0, d_conv1_0, d_pool1_0, d_conv2_0, d_pool2_0, d_fc1_0, d_fc1relu_0, d_fc2_0, d_fc2smax_0, d_pconv1_0, d_pconv1bias_0, d_pconv2_0, d_pconv2bias_0, d_pfc1_0, d_pfc1bias_0,
                                            d_pfc2_0, d_pfc2bias_0, d_cudnn_workspace_0, d_onevec_0);

            // Perform classification
            			std::vector<float> class_vec(10);

            // Copy back result
            			checkCudaErrors(cudaMemcpy(&class_vec[0], d_fc2smax_0, sizeof(float) * 10, cudaMemcpyDeviceToHost));

            // Determine classification according to maximal response
            			int chosen = 0;
            			for (int id = 1; id < 10; ++id)
            			{	
                			if (class_vec[chosen] < class_vec[id]) chosen = id;
            			}

            		if (chosen != test_labels[i])
                		++num_errors;
        		}
        		classification_error = (float)num_errors / (float)classifications;

        		printf("Classification result: %.2f%% error (used %d images)\n", classification_error * 100.0f, (int)classifications);
       
    		}
  // Free data structures
/*    		checkCudaErrors(cudaFree(d_data));
    		checkCudaErrors(cudaFree(d_conv1));
    		checkCudaErrors(cudaFree(d_pool1));
    		checkCudaErrors(cudaFree(d_conv2));
    		checkCudaErrors(cudaFree(d_pool2));
    		checkCudaErrors(cudaFree(d_fc1));
    		checkCudaErrors(cudaFree(d_fc2));
    		checkCudaErrors(cudaFree(d_pconv1));
    		checkCudaErrors(cudaFree(d_pconv1bias));
    		checkCudaErrors(cudaFree(d_pconv2));
    		checkCudaErrors(cudaFree(d_pconv2bias));
    		checkCudaErrors(cudaFree(d_pfc1));
    		checkCudaErrors(cudaFree(d_pfc1bias));
    		checkCudaErrors(cudaFree(d_pfc2));
    		checkCudaErrors(cudaFree(d_pfc2bias));
    		checkCudaErrors(cudaFree(d_gconv1));
    		checkCudaErrors(cudaFree(d_gconv1bias));
    		checkCudaErrors(cudaFree(d_gconv2));
    		checkCudaErrors(cudaFree(d_gconv2bias));
    		checkCudaErrors(cudaFree(d_gfc1));
    		checkCudaErrors(cudaFree(d_gfc1bias));
    		checkCudaErrors(cudaFree(d_dfc1));
    		checkCudaErrors(cudaFree(d_gfc2));
    		checkCudaErrors(cudaFree(d_gfc2bias));
    		checkCudaErrors(cudaFree(d_dfc2));
    		checkCudaErrors(cudaFree(d_dpool1));
    		checkCudaErrors(cudaFree(d_dconv2));
    		checkCudaErrors(cudaFree(d_dpool2));    
    		checkCudaErrors(cudaFree(d_labels));
    		checkCudaErrors(cudaFree(d_dlossdata));
    		checkCudaErrors(cudaFree(d_onevec));
    		if (d_cudnn_workspace != nullptr)
        		checkCudaErrors(cudaFree(d_cudnn_workspace));
	*/
    		return 0;
		}
}
