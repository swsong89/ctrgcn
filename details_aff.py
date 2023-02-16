Model(
  (data_bn): BatchNorm1d(150, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (l1): TCN_GCN_unit(
    (gcn1): unit_gcn(
      (convs): ModuleList(
        (0): CTRGC(
          (conv1): Conv2d(3, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(3, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(3, 64, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(8, 64, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
        (1): CTRGC(
          (conv1): Conv2d(3, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(3, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(3, 64, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(8, 64, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
        (2): CTRGC(
          (conv1): Conv2d(3, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(3, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(3, 64, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(8, 64, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
      )
      (down): Sequential(
        (0): Conv2d(3, 64, kernel_size=(1, 1), stride=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (soft): Softmax(dim=-2)
      (relu): ReLU(inplace=True)
    )
    (tcn1): MultiScale_TemporalConv(
      (branches): ModuleList(
        (0): Sequential(
          (0): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): TemporalConv(
            (conv): Conv2d(16, 16, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
            (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Sequential(
          (0): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): TemporalConv(
            (conv): Conv2d(16, 16, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), dilation=(2, 1))
            (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (2): Sequential(
          (0): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), dilation=1, ceil_mode=False)
          (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (3): Sequential(
          (0): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (aff): AFF(
        (local_att): Sequential(
          (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (global_att): Sequential(
          (0): AdaptiveAvgPool2d(output_size=1)
          (1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(inplace=True)
          (4): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (sigmoid): Sigmoid()
      )
    )
    (relu): ReLU(inplace=True)
  )
  (l2): TCN_GCN_unit(
    (gcn1): unit_gcn(
      (convs): ModuleList(
        (0): CTRGC(
          (conv1): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(8, 64, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
        (1): CTRGC(
          (conv1): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(8, 64, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
        (2): CTRGC(
          (conv1): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(8, 64, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
      )
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (soft): Softmax(dim=-2)
      (relu): ReLU(inplace=True)
    )
    (tcn1): MultiScale_TemporalConv(
      (branches): ModuleList(
        (0): Sequential(
          (0): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): TemporalConv(
            (conv): Conv2d(16, 16, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
            (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Sequential(
          (0): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): TemporalConv(
            (conv): Conv2d(16, 16, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), dilation=(2, 1))
            (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (2): Sequential(
          (0): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), dilation=1, ceil_mode=False)
          (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (3): Sequential(
          (0): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (aff): AFF(
        (local_att): Sequential(
          (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (global_att): Sequential(
          (0): AdaptiveAvgPool2d(output_size=1)
          (1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(inplace=True)
          (4): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (sigmoid): Sigmoid()
      )
    )
    (relu): ReLU(inplace=True)
  )
  (l3): TCN_GCN_unit(
    (gcn1): unit_gcn(
      (convs): ModuleList(
        (0): CTRGC(
          (conv1): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(8, 64, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
        (1): CTRGC(
          (conv1): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(8, 64, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
        (2): CTRGC(
          (conv1): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(8, 64, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
      )
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (soft): Softmax(dim=-2)
      (relu): ReLU(inplace=True)
    )
    (tcn1): MultiScale_TemporalConv(
      (branches): ModuleList(
        (0): Sequential(
          (0): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): TemporalConv(
            (conv): Conv2d(16, 16, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
            (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Sequential(
          (0): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): TemporalConv(
            (conv): Conv2d(16, 16, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), dilation=(2, 1))
            (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (2): Sequential(
          (0): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), dilation=1, ceil_mode=False)
          (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (3): Sequential(
          (0): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (aff): AFF(
        (local_att): Sequential(
          (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (global_att): Sequential(
          (0): AdaptiveAvgPool2d(output_size=1)
          (1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(inplace=True)
          (4): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (sigmoid): Sigmoid()
      )
    )
    (relu): ReLU(inplace=True)
  )
  (l4): TCN_GCN_unit(
    (gcn1): unit_gcn(
      (convs): ModuleList(
        (0): CTRGC(
          (conv1): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(8, 64, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
        (1): CTRGC(
          (conv1): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(8, 64, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
        (2): CTRGC(
          (conv1): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(8, 64, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
      )
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (soft): Softmax(dim=-2)
      (relu): ReLU(inplace=True)
    )
    (tcn1): MultiScale_TemporalConv(
      (branches): ModuleList(
        (0): Sequential(
          (0): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): TemporalConv(
            (conv): Conv2d(16, 16, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
            (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Sequential(
          (0): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): TemporalConv(
            (conv): Conv2d(16, 16, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), dilation=(2, 1))
            (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (2): Sequential(
          (0): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), dilation=1, ceil_mode=False)
          (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (3): Sequential(
          (0): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (aff): AFF(
        (local_att): Sequential(
          (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (global_att): Sequential(
          (0): AdaptiveAvgPool2d(output_size=1)
          (1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(inplace=True)
          (4): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (sigmoid): Sigmoid()
      )
    )
    (relu): ReLU(inplace=True)
  )
  (l5): TCN_GCN_unit(
    (gcn1): unit_gcn(
      (convs): ModuleList(
        (0): CTRGC(
          (conv1): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(8, 128, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
        (1): CTRGC(
          (conv1): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(8, 128, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
        (2): CTRGC(
          (conv1): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(8, 128, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
      )
      (down): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (soft): Softmax(dim=-2)
      (relu): ReLU(inplace=True)
    )
    (tcn1): MultiScale_TemporalConv(
      (branches): ModuleList(
        (0): Sequential(
          (0): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): TemporalConv(
            (conv): Conv2d(32, 32, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0))
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Sequential(
          (0): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): TemporalConv(
            (conv): Conv2d(32, 32, kernel_size=(5, 1), stride=(2, 1), padding=(4, 0), dilation=(2, 1))
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (2): Sequential(
          (0): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dilation=1, ceil_mode=False)
          (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (3): Sequential(
          (0): Conv2d(128, 32, kernel_size=(1, 1), stride=(2, 1))
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (aff): AFF(
        (local_att): Sequential(
          (0): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (global_att): Sequential(
          (0): AdaptiveAvgPool2d(output_size=1)
          (1): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(inplace=True)
          (4): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (sigmoid): Sigmoid()
      )
    )
    (relu): ReLU(inplace=True)
    (residual): unit_tcn(
      (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 1))
      (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (l6): TCN_GCN_unit(
    (gcn1): unit_gcn(
      (convs): ModuleList(
        (0): CTRGC(
          (conv1): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(16, 128, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
        (1): CTRGC(
          (conv1): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(16, 128, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
        (2): CTRGC(
          (conv1): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(16, 128, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
      )
      (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (soft): Softmax(dim=-2)
      (relu): ReLU(inplace=True)
    )
    (tcn1): MultiScale_TemporalConv(
      (branches): ModuleList(
        (0): Sequential(
          (0): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): TemporalConv(
            (conv): Conv2d(32, 32, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Sequential(
          (0): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): TemporalConv(
            (conv): Conv2d(32, 32, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), dilation=(2, 1))
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (2): Sequential(
          (0): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), dilation=1, ceil_mode=False)
          (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (3): Sequential(
          (0): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (aff): AFF(
        (local_att): Sequential(
          (0): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (global_att): Sequential(
          (0): AdaptiveAvgPool2d(output_size=1)
          (1): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(inplace=True)
          (4): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (sigmoid): Sigmoid()
      )
    )
    (relu): ReLU(inplace=True)
  )
  (l7): TCN_GCN_unit(
    (gcn1): unit_gcn(
      (convs): ModuleList(
        (0): CTRGC(
          (conv1): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(16, 128, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
        (1): CTRGC(
          (conv1): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(16, 128, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
        (2): CTRGC(
          (conv1): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(16, 128, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
      )
      (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (soft): Softmax(dim=-2)
      (relu): ReLU(inplace=True)
    )
    (tcn1): MultiScale_TemporalConv(
      (branches): ModuleList(
        (0): Sequential(
          (0): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): TemporalConv(
            (conv): Conv2d(32, 32, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Sequential(
          (0): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): TemporalConv(
            (conv): Conv2d(32, 32, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), dilation=(2, 1))
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (2): Sequential(
          (0): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), dilation=1, ceil_mode=False)
          (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (3): Sequential(
          (0): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (aff): AFF(
        (local_att): Sequential(
          (0): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (global_att): Sequential(
          (0): AdaptiveAvgPool2d(output_size=1)
          (1): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(inplace=True)
          (4): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (sigmoid): Sigmoid()
      )
    )
    (relu): ReLU(inplace=True)
  )
  (l8): TCN_GCN_unit(
    (gcn1): unit_gcn(
      (convs): ModuleList(
        (0): CTRGC(
          (conv1): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(16, 256, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
        (1): CTRGC(
          (conv1): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(16, 256, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
        (2): CTRGC(
          (conv1): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(16, 256, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
      )
      (down): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (soft): Softmax(dim=-2)
      (relu): ReLU(inplace=True)
    )
    (tcn1): MultiScale_TemporalConv(
      (branches): ModuleList(
        (0): Sequential(
          (0): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): TemporalConv(
            (conv): Conv2d(64, 64, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0))
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Sequential(
          (0): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): TemporalConv(
            (conv): Conv2d(64, 64, kernel_size=(5, 1), stride=(2, 1), padding=(4, 0), dilation=(2, 1))
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (2): Sequential(
          (0): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dilation=1, ceil_mode=False)
          (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (3): Sequential(
          (0): Conv2d(256, 64, kernel_size=(1, 1), stride=(2, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (aff): AFF(
        (local_att): Sequential(
          (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (global_att): Sequential(
          (0): AdaptiveAvgPool2d(output_size=1)
          (1): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(inplace=True)
          (4): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (5): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (sigmoid): Sigmoid()
      )
    )
    (relu): ReLU(inplace=True)
    (residual): unit_tcn(
      (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 1))
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (l9): TCN_GCN_unit(
    (gcn1): unit_gcn(
      (convs): ModuleList(
        (0): CTRGC(
          (conv1): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(32, 256, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
        (1): CTRGC(
          (conv1): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(32, 256, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
        (2): CTRGC(
          (conv1): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(32, 256, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
      )
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (soft): Softmax(dim=-2)
      (relu): ReLU(inplace=True)
    )
    (tcn1): MultiScale_TemporalConv(
      (branches): ModuleList(
        (0): Sequential(
          (0): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): TemporalConv(
            (conv): Conv2d(64, 64, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Sequential(
          (0): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): TemporalConv(
            (conv): Conv2d(64, 64, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), dilation=(2, 1))
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (2): Sequential(
          (0): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), dilation=1, ceil_mode=False)
          (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (3): Sequential(
          (0): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (aff): AFF(
        (local_att): Sequential(
          (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (global_att): Sequential(
          (0): AdaptiveAvgPool2d(output_size=1)
          (1): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(inplace=True)
          (4): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (5): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (sigmoid): Sigmoid()
      )
    )
    (relu): ReLU(inplace=True)
  )
  (l10): TCN_GCN_unit(
    (gcn1): unit_gcn(
      (convs): ModuleList(
        (0): CTRGC(
          (conv1): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(32, 256, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
        (1): CTRGC(
          (conv1): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(32, 256, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
        (2): CTRGC(
          (conv1): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
          (conv2): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
          (conv3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (conv4): Conv2d(32, 256, kernel_size=(1, 1), stride=(1, 1))
          (tanh): Tanh()
        )
      )
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (soft): Softmax(dim=-2)
      (relu): ReLU(inplace=True)
    )
    (tcn1): MultiScale_TemporalConv(
      (branches): ModuleList(
        (0): Sequential(
          (0): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): TemporalConv(
            (conv): Conv2d(64, 64, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Sequential(
          (0): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): TemporalConv(
            (conv): Conv2d(64, 64, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), dilation=(2, 1))
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (2): Sequential(
          (0): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), dilation=1, ceil_mode=False)
          (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (3): Sequential(
          (0): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (aff): AFF(
        (local_att): Sequential(
          (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (global_att): Sequential(
          (0): AdaptiveAvgPool2d(output_size=1)
          (1): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(inplace=True)
          (4): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (5): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (sigmoid): Sigmoid()
      )
    )
    (relu): ReLU(inplace=True)
  )
  (fc): Linear(in_features=256, out_features=60, bias=True)
)