class arg(object):
    def __init__(self, prefix="", evl=False, batchSize=128, inputLen=10, outputLen=90, 
                 attn=False, inputDim=3, outputDim=3, hiddenSize=64, epochs=10,
                 lr=0.001, momentum=0.5, cuda=True, seed=1, fromCkpt="", l2=0.1, opt="sgd", device=0, jankTrain=False):
        self.prefix=prefix
        self.evl=evl
        self.batch_size=batchSize
        self.input_len=inputLen
        self.output_len=outputLen
        self.attn=attn
        self.state_dim=inputDim
        self.state_dim_out=outputDim
        self.hidden_size=hiddenSize
        self.epochs=epochs
        self.lr=lr
        self.momentum=momentum
        self.cuda=cuda
        self.seed=seed
        self.from_ckpt=fromCkpt
        self.l2=l2
        self.opt=opt
        self.device=device
        self.jankTrain=jankTrain
