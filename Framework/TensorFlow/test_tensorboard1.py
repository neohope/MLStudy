#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
tensorboard --logdir="PATH_TO_OUTPUT/output/improved_graph"
"""

import tensorflow as tf


def init_graph(graph):
    with graph.as_default():
        with tf.name_scope("variables"):
            # 运行次数
            global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
            # 输出数值总计
            total_output = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="total_output")

        with tf.name_scope("transformation"):
            # 输入层
            with tf.name_scope("input"):
                # Create input placeholder- takes in a Vector
                a = tf.placeholder(tf.float32, shape=[None], name="input_placeholder_a")

            # 中间层
            with tf.name_scope("intermediate_layer"):
                b = tf.reduce_prod(a, name="product_b")
                c = tf.reduce_sum(a, name="sum_c")

            # 输出层
            with tf.name_scope("output"):
                output = tf.add(b, c, name="output")

        with tf.name_scope("update"):
            # 更新运行次数以及数值总计
            update_total = total_output.assign_add(output)
            increment_step = global_step.assign_add(1)

        with tf.name_scope("summaries"):
            avg = tf.div(update_total, tf.cast(increment_step, tf.float32), name="average")
            tf.summary.scalar("output_summary", output)
            tf.summary.scalar("total_summary", update_total)
            tf.summary.scalar("average_summary", avg)

        with tf.name_scope("global_ops"):
            init = tf.global_variables_initializer()
            merged_summaries = tf.summary.merge_all()

    return a, output, increment_step, init, merged_summaries


def run_graph(sess, writer, a, input_tensor, output, increment_step, merged_summaries):
    """
    通过给定输入，计算并保存结果
    """
    feed_dict = {a: input_tensor}
    _, step, summary = sess.run([output, increment_step, merged_summaries], feed_dict=feed_dict)
    writer.add_summary(summary, global_step=step)


if __name__ == '__main__':
    # 初始化graph
    graph = tf.Graph()
    a, output, increment_step, init, merged_summaries = init_graph(graph)

    # 新建session
    sess = tf.Session(graph=graph)

    # Open a SummaryWriter to save summaries
    writer = tf.summary.FileWriter('output/improved_graph', graph)

    # 初始化
    sess.run(init)

    # 用多个输入，运行程序
    run_graph(sess, writer, a, [2,8], output, increment_step, merged_summaries)
    run_graph(sess, writer, a, [3,1,3,3], output, increment_step, merged_summaries)
    run_graph(sess, writer, a, [8], output, increment_step, merged_summaries)
    run_graph(sess, writer, a, [1,2,3], output, increment_step, merged_summaries)
    run_graph(sess, writer, a, [11,4], output, increment_step, merged_summaries)
    run_graph(sess, writer, a, [4,1], output, increment_step, merged_summaries)
    run_graph(sess, writer, a, [7,3,1], output, increment_step, merged_summaries)
    run_graph(sess, writer, a, [6,3], output, increment_step, merged_summaries)
    run_graph(sess, writer, a, [0,2], output, increment_step, merged_summaries)
    run_graph(sess, writer, a, [4,5,6], output, increment_step, merged_summaries)

    # 写入结果
    writer.flush()
    writer.close()
    sess.close()
