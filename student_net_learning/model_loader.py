

def get_model_net(model_name):
    from student_net_learning.models.densenet import densenet201
    print('Loading DenseNet121')
    net = densenet201(pretrained=True)
    return net