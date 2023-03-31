

def calculate_linear_layer_size_after_conv(input_shape, kernel_size, stride, padding):
    '''
    Calculates the size of the linear layer after a convolutional layer.
    
    input_shape: tuple of ints (channels, height, width)
    '''
    channels, height, width = input_shape
    height = (height - kernel_size + 2*padding) / stride + 1
    width = (width - kernel_size + 2*padding) / stride + 1
    return int(channels * height * width)