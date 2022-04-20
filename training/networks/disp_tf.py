def disparity_refinement(features, disp):
    conv1a = features[0]
    up_conv1a2a = features[4]
    disp0 = disp[0]
    disp1 = disp[1]
    disp2 = disp[2]

    disp0_a = tf.expand_dims(disp0[:, :, :, 0], 3)
    conv1b = generate_image_right(conv1a, tf.expand_dims(disp1[:, :, :, 1], 3))
    up_conv1b2b = generate_image_right(up_conv1a2a, tf.expand_dims(disp0[:, :, :, 1], 3))

    print(' - Disparity Refinement Sub-network')

    with tf.variable_scope("w_up_conv1b2b"):
        w_up_conv1b2b = generate_image_left(up_conv1b2b, disp0_a)
        print ('w_up_conv1b2b:')
        print (w_up_conv1b2b.get_shape().as_list())

    with tf.variable_scope("r_conv0"):
        r_conv0 = conv2d(tf.concat([tf.abs(up_conv1a2a - w_up_conv1b2b), disp0, up_conv1a2a], axis=3),
                         [3, 3, 66, 32], 1, True)
        print ('r_conv0:')
        print (r_conv0.get_shape().as_list())

    with tf.variable_scope("r_conv1"):
        r_conv1 = conv2d(r_conv0, [3, 3, 32, 64], 2, True)
        print ('r_conv1:')
        print (r_conv1.get_shape().as_list())

    with tf.variable_scope("c_conv1a"):
        c_conv1a = conv2d(conv1a, [3, 3, 64, 16], 1, True)
        print ('c_conv1a:')
        print (c_conv1a.get_shape().as_list())

    with tf.variable_scope("c_conv1b"):
        c_conv1b = conv2d(conv1b, [3, 3, 64, 16], 1, True)
        print ('c_conv1b:')
        print (c_conv1b.get_shape().as_list())

    with tf.variable_scope("r_corr"):
        r_corr = correlation_map(c_conv1a, c_conv1b, 20)
        print ('r_corr:')
        print (r_corr.get_shape().as_list())

    with tf.variable_scope("r_conv1_1"):
        r_conv1_1 = conv2d(tf.concat([r_corr, r_conv1], axis=3), [3, 3, 105, 64], 1, True)
        print ('r_conv1_1:')
        print (r_conv1_1.get_shape().as_list())

    with tf.variable_scope("r_conv2"):
        r_conv2 = conv2d(r_conv1_1, [3, 3, 64, 128], 2, True)
        print ('r_conv2:')
        print (r_conv2.get_shape().as_list())

    with tf.variable_scope("r_conv2_1"):
        r_conv2_1 = conv2d(r_conv2, [3, 3, 128, 128], 1, True)
        print ('r_conv2_1:')
        print (r_conv2_1.get_shape().as_list())

    with tf.variable_scope("r_res2"):
        r_res2 = conv2d(tf.concat([r_conv2_1, disp2], axis=3), [3, 3, 130, 1], 1, True)
        print ('r_res2:')
        print (r_res2.get_shape().as_list())

    with tf.variable_scope("r_upconv1"):
        r_upconv1 = conv2d_transpose(r_conv2_1, [4, 4, 64, 128], 2, True)
        print ('r_upconv1:')
        print (r_upconv1.get_shape().as_list())

    with tf.variable_scope("r_iconv1"):
        r_iconv1 = conv2d(tf.concat([r_upconv1, conv2d_transpose(r_res2, [4, 4, 2, 1], 2, True), r_conv1_1],
                                    axis=3), [3, 3, 130, 64], 1, True)
        print ('r_iconv1:')
        print (r_iconv1.get_shape().as_list())

    with tf.variable_scope("r_res1"):
        r_res1 = conv2d(tf.concat([r_iconv1, disp1], axis=3), [3, 3, 66, 1], 1, True)
        print ('r_res1:')
        print (r_res1.get_shape().as_list())

    with tf.variable_scope("r_upconv0"):
        r_upconv0 = conv2d_transpose(r_iconv1, [4, 4, 32, 64], 2, True)
        print ('r_upconv0:')
        print (r_upconv0.get_shape().as_list())

    with tf.variable_scope("r_iconv0"):
        r_iconv0 = conv2d(tf.concat([r_upconv0, conv2d_transpose(r_res1, [4, 4, 2, 1], 2, True), r_conv0],
                                    axis=3), [3, 3, 66, 32], 1, True)
        print ('r_iconv0:')
        print (r_iconv0.get_shape().as_list())

    with tf.variable_scope("r_res0"):
        r_res0 = conv2d(tf.concat([r_iconv0, disp0], axis=3), [3, 3, 34, 1], 1, True)
        print ('r_res0:')
        print (r_res0.get_shape().as_list())

    return r_res0, r_res1, r_res2
