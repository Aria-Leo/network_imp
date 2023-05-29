import numpy as np


class Functional:

    @staticmethod
    def sigmoid(z):
        res = 1.0 / (1.0 + np.exp(-z))
        return res

    @staticmethod
    def sigmoid_prime(z):
        return Functional.sigmoid(z) * (1 - Functional.sigmoid(z))

    @staticmethod
    def relu(z):
        return np.where(z > 0, z, 0)

    @staticmethod
    def relu_prime(z):
        return np.where(z > 0, 1, 0)

    @staticmethod
    def leaky_relu(z, decay=0.1):
        return np.where(z > 0, z, decay * z)

    @staticmethod
    def leaky_relu_prime(z, decay=0.1):
        return np.where(z > 0, 1, decay)

    @staticmethod
    def softmax(z):
        """

        Args:
            z: 1-D or 2-D array

        Returns:

        """
        z_dim = len(z.shape)
        if z_dim == 1:
            z = z.reshape(1, -1)
        e_z = np.exp(z)
        e_sum = np.sum(e_z, axis=1)
        res = np.divide(e_z.T, e_sum).T
        if z_dim == 1:
            res = res.flatten()
        return res

    @staticmethod
    def softmax_prime(z):
        """

        Args:
            z: 1-D or 2-D array

        Returns:

        """
        def base_diff(e):
            """

            Args:
                e: 1-D array

            Returns:

            """
            return np.diag(e) - e.reshape(-1, 1) @ e.reshape(1, -1)

        y = Functional.softmax(z)
        z_dim = len(z.shape)
        if z_dim == 1:
            res = base_diff(y)
        else:
            res = np.apply_along_axis(base_diff, 1, y)
        return res

    @staticmethod
    def add_dilation(input_, dilated_step=1):
        """
        给input_相邻元素添加空洞
        Args:
            input_: 2-D np.array
            dilated_step: 相邻元素添加几个空洞

        Returns:

        """
        stride = dilated_step + 1
        i_list = [i * stride for i in range(input_.shape[0]) for _ in range(input_.shape[1])]
        j_list = [j * stride for _ in range(input_.shape[0]) for j in range(input_.shape[1])]
        res = np.zeros(((input_.shape[0] - 1) * stride + 1, (input_.shape[1] - 1) * stride + 1))
        res[i_list, j_list] = input_.flatten()
        return res

    @staticmethod
    def conv2d(input_, kernel, stride=1, padding='same',
               conv_mode='normal', dilated_feature=0, dilated_kernel=0):
        """
        定义一下计算规则：
        1.input_->(h_in, w_in), kernel->(kernel_size, kernel_size):
            表示kernel会与input_做卷积，输出结果的shape为(h_out, w_out)
        2.input_->(c, h_in, w_in), kernel->(kernel_size, kernel_size):
            表示kernel会与input_的每个channel分别做卷积，输出结果的shape为(c, h_out, w_out)
        3.input_->(h_in, w_in), kernel->(c, kernel_size, kernel_size):
            表示kernel的每个channel都会与input_卷积，输出结果为(c, h_out, w_out)
        4.input_->(c, h_in, w_in), kernel->(c, kernel_size, kernel_size):
            这里input_和kernel的通道数必须相同，表示相同通道位置的single_input和single_filter做卷积，
            输出结果为(c, h_out, w_out)
        Args:
            input_: np.array, shape: (c, h_in, w_in) or (h_in, w_in)
            kernel: np.array, shape: (c, kernel_size, kernel_size) or (kernel_size, kernel_size)
            stride:
            padding:
            conv_mode: 'normal' or 'math'. normal表示深度学习中的卷积，math表示数学中的卷积，
                此时需要将kernel每个channel旋转180度
            dilated_feature: 对input_相邻元素之间添加几个空洞，默认不添加
                举例: dilated=1:
                     [1 2 3         [1 0 2 0 3
                      4 5 6          0 0 0 0 0
                      7 8 9]   ->    4 0 5 0 6
                                     0 0 0 0 0
                                     7 0 8 0 9]
            dilated_kernel: 对kernel相邻元素添加空洞，含义与dilated_feature类似

        Returns:

        """
        kernel_size = kernel.shape[0]
        if padding == 'same':
            padding_size = kernel_size // 2
        elif padding == 'valid':
            padding_size = 0
        else:
            padding_size = padding
        res = []
        input_shape_len = len(input_.shape)
        kernel_shape_len = len(kernel.shape)
        if input_shape_len == kernel_shape_len:
            if input_shape_len == 2:
                input_ = np.expand_dims(input_, axis=0)
                kernel = np.expand_dims(kernel, axis=0)
        else:
            if input_shape_len == 2:
                input_ = (input_ for _ in range(kernel.shape[0]))
            else:
                kernel = (kernel for _ in range(input_.shape[0]))
        for s_input, s_kernel in zip(input_, kernel):
            if conv_mode == 'math':
                s_kernel = np.rot90(s_kernel, 2)
            if dilated_kernel > 0:
                s_kernel = Function.add_dilation(s_kernel, dilated_kernel)
            if dilated_feature > 0:
                s_input = Function.add_dilation(s_input, dilated_feature)
            s_input = np.pad(s_input, pad_width=padding_size)
            result = []
            out_shape_h = (s_input.shape[0] - kernel_size) // stride + 1
            out_shape_w = (s_input.shape[1] - kernel_size) // stride + 1
            for i in range(out_shape_h):
                for j in range(out_shape_w):
                    temp_res = np.sum(s_input[i * stride: i * stride + kernel_size,
                                      j * stride: j * stride + kernel_size]
                                      * s_kernel)
                    result.append(temp_res)
            result = np.reshape(result, (out_shape_h, out_shape_w))
            res.append(result)
        res = np.array(res)
        return res
