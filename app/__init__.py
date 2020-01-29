#--------ALI AKBAR MAHBADI :aamahbadi@yahoo.com----------
from flask import Flask, render_template
import csv, os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import BytesIO
import base64
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import preprocessing


app = Flask(__name__,
            static_url_path=''
            )


from app import views
