# The code shows how to implement Variational Auto-Encoder in Julia
# MIT license

using Flux, Random
using Flux: logitbinarycrossentropy, chunk
using Flux.Optimise
using Flux.Data
using Flux: throttle
using Statistics
using Images

x_dim = 784
h_dim = 256
z_dim = 5
layers = Array{Dense}(undef, 5)

batch_size = 100
sample_size = 10
data_dir = "<local directory for mnist dataset>"
project_dir = "<local directory for vae project>"
train_data_filename = "train-images.idx3-ubyte"

# Load MNIST training data from local file system. It's to circumvent the situation
# where downloading MNIST may be rather slow.
function load_images(dir)
    filepath = joinpath(dir, train_data_filename)
    io = IOBuffer(read(filepath))
    _, N, nrows, ncols = MNIST.imageheader(io)
    images = [MNIST.rawimage(io) for _ in 1:N]
    # Transform into (784, N) in shape and change each data element into Float32
    return Float32.(hcat(vec.(images)...))
end
images = load_images(data_dir)
nobs = size(images, 2)
# Partition the whole dataset into a group of mini-batches
data = [images[:, i] for i in Iterators.partition(1:nobs, batch_size)]

# Change a mini-batch of data into 2-dimensional gray image 
# and save the image as a file
function save_image(batch_data, filename)
    chunked_data = chunk(batch_data, sample_size)
    im_data = reshape.(chunked_data, 28, :)
    im = Gray.(vcat(im_data...))
    image_path = joinpath(project_dir, filename)
    save(image_path, im)
end

# Take a sample of mini-batch as the reconstruction target 
sample_data = data[1]
save_image(sample_data, "mnist_base.png")

# Encoder network
layers[1] = Dense(x_dim, h_dim, relu)
layers[2] = Dense(h_dim, z_dim)
layers[3] = Dense(h_dim, z_dim)

# Encoder network has branch-out instead of sequential topology 
function g(x)
    h = layers[1](x)
    return (layers[2](h), layers[3](h))
end

function z(mu, logsig)
    sigma = exp.(logsig / 2)
    return mu + randn(Float32) .* sigma
end

# Decoder network has sequential topology
layers[4] = Dense(z_dim, h_dim, relu)
layers[5] = Dense(h_dim, x_dim)
decode(z) = Chain(layers[4], layers[5])(z)

# KL loss
loss_kl(mu, logsig) = 0.5 * sum(mu .^ 2 + exp.(logsig) - logsig .- 1, dims=1)

# Reconstruction loss
loss_reconstruct(x, z) = sum(logitbinarycrossentropy.(decode(z), x), dims=1)

# Loss function comprises of KL loss and reconstruction loss
function loss(x)
    miu, logsig = g(x)
    encoded_z = z(miu, logsig)
    kl = loss_kl(miu, logsig)
    rec = loss_reconstruct(x, encoded_z)
    mean(kl + rec)
end

# Reconstruct images given the input sample
function reconstruct(x)
    miu, logsig = g(x)
    encoded_z = z(miu, logsig)
    decoded_x = decode(encoded_z)
    sigmoid.(decoded_x)
end

opt = ADAM()
ps = Flux.params(layers[1:5]...)

# Compute loss against a random batch every 30 seconds
evalcb = throttle(() -> @show(loss(images[:, rand(1:nobs, batch_size)])), 30)

for epoch in 1:50
    @info "Epoch $epoch"
    train!(loss, ps, zip(data), opt, cb=evalcb)

    # Reconstruct the sample image and save the reconstructed image after each epoch of training
    decoded_sample = reconstruct(sample_data)
    save_image(decoded_sample, "decode_sample_$epoch.png")
end
