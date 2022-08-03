
from PIL import Image
import os, os.path
import numpy as np
import pandas as pd
from pde import PDEBase, ScalarField, UnitGrid, CartesianGrid


def load_chinese(path_chars,path_csv):
    #Thanks to Richard Kuo
    #https://www.kaggle.com/code/rkuo2000/chinese-mnist

    imgs = []
    labels = []
    path = path_chars + "chars"
    valid_images = [".jpg",".gif",".png",".tga"]
    for f in os.listdir(path)[:]:
        labels.append(int(f.split('_')[-1][:-4]))
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        imgs.append(np.ndarray.flatten(np.array(Image.open(os.path.join(path,f)))))

    imgs = np.asarray(imgs)
    df = pd.read_csv(path_csv + "chinese_mnist.csv")
    def file_path_col(df):    
        file_path = f"input_{df[0]}_{df[1]}_{df[2]}.jpg" #input_1_1_10.jpg    
        return file_path

    df["file_path"] = df.apply(file_path_col, axis = 1)
    import skimage.io
    import skimage.transform

    file_paths = list(df.file_path)
    def read_image(file_paths):
        image = skimage.io.imread("C:/Users/Elchanan/Desktop/Work/convperstran/data/chars/" + file_paths)
        image = skimage.transform.resize(image, (64, 64), mode="reflect") 
        # THe mode parameter determines how the array borders are handled.    
        return image[:, :]

    # One hot encoder, but in 15 classes
    def character_encoder(df, var = "character"):
        x = np.stack(df["file_path"].apply(read_image))
        y = pd.get_dummies(df[var], drop_first = False)
        return x, y

    X,y = character_encoder(df)
    y = df['value']
    return X,y.to_numpy()

def KS_dataset(n,r_range,t=15):
    X = []
    y = []
    for r in r_range:
        surfaces = generate_KS_surfaces(n,r,t)
        X += surfaces
        y+= [r]*n
    return X,y


   

def generate_KS_surfaces(n,r,t):
    surfaces = []

    class KuramotoSivashinskyPDE(PDEBase):
        def evolution_rate(self, state,t0=0):
            state_lap = state.laplace(bc="auto_periodic_neumann")
            state_lap2 = state_lap.laplace(bc="auto_periodic_neumann")
            state_grad_x = state.gradient(bc="auto_periodic_neumann")[0]
            state_grad_y = state.gradient(bc="auto_periodic_neumann")[1]
            return r*state_grad_x.to_scalar()**2 + state_grad_y.to_scalar()**2  - state_lap - state_lap2

    
    grid = UnitGrid([50,50])  # generate grid
    #grid = CartesianGrid([[-30, 30], [-30, 30]], 75)
    state = ScalarField.random_uniform(grid)  # generate initial condition
    
    for i in range(n):
        state = ScalarField.random_uniform(grid)  # generate initial condition
        eq  = KuramotoSivashinskyPDE()
        result = eq.solve(state, t, dt=0.01)
        surfaces.append(np.asarray(result.data))

    return surfaces