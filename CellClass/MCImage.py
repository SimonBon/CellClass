import numpy as np

class MCImage():

    def __init__(self, img, scheme="BGR"):
        
        scheme = scheme.upper()

        if scheme not in ["BGR", "RGB"]:
            raise ValueError("Not a valid color scheme! Please use: 'BGR', 'RGB'")
        else:

            if scheme == "BGR":
                self.B = img[..., 0]
                self.G = img[..., 1]
                self.R = img[..., 2]

            elif scheme == "RGB":
                self.B = img[..., 2]
                self.G = img[..., 1]
                self.R = img[..., 0]
                

            self.RGB = np.stack((self.R, self.G, self.B), axis=-1)

    def normalize(self, channelwise=True):
        
        if channelwise:
            self.B = (self.B-self.B.min())/(self.B.max()-self.B.min())
            self.R = (self.R-self.R.min())/(self.R.max()-self.R.min())
            self.G = (self.G-self.G.min())/(self.G.max()-self.G.min())

        else:
            stack = np.stack((self.B, self.G, self.R), axis=-1)
            stack = (stack-stack.min())/(stack.max()-stack.min())
            self.B = stack[..., 0]
            self.G = stack[..., 1]
            self.R = stack[..., 2]

    
