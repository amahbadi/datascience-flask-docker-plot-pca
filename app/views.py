#--------ALI AKBAR MAHBADI :aamahbadi@yahoo.com----------
from app import app, render_template, csv, os, plt, pd, np, BytesIO, base64, Axes3D, PCA, preprocessing


count = 0
df = pd.read_csv('static/titanic.csv')
def readcsv(filename, count):
    for row in open(os.path.join(os.path.dirname(__file__), filename), 'r'):
        yield row
        count += 1
        if count == 15:
            break

FILE = 'static/titanic.csv'
iter_file = iter(readcsv(FILE, count))

list_data = pd.read_csv(FILE)

def get_plot_data(field_name, list_data):
    field = list(np.array(list_data[field_name]))
    mu , sigma = np.mean(field), np.std(field)
    plt.figure()
    plt.hist(field, 50, density=True, facecolor='#FF1744', alpha=0.75)
    plt.xlabel(field_name)
    plt.ylabel('Probability')
    plt.title('Histogram of %s' %(field_name))
    plt.text(48, .03, r'$\mu={},\ \sigma={}$'.format(round(mu),round(sigma)))
    plt.grid(color='k', linestyle='--', linewidth=0.5)
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png

def get_plot_pca(df):
    df.Age = df.Age.fillna(-0.5)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, (-5, 0, 5, 15, 25, 35, 45, 55, 80), labels=group_names)
    df.Age = categories
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    df.Fare = df.Fare.fillna(-0.5)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, (-5, 0, 5, 15, 35, 1000), labels=group_names)
    df.Fare = categories
    data=df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

    features = ['Fare', 'Cabin', 'Age', 'Sex']
        
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(data[feature])
        data[feature] = le.transform(data[feature])

    scaler = preprocessing.StandardScaler()
    scaler.fit(data)

    preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
    scaled_data = scaler.transform(data)

    pca = PCA(n_components=3)
    pca.fit(scaled_data)
    x_pca = pca.transform(scaled_data)
    X = x_pca[:,0]
    Y = x_pca[:,1]
    Z = x_pca[:,2]

    features = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']
    for feature in features:
        fig = plt.figure(figsize=(6,4), dpi=95)
        ax = Axes3D(fig)
        p = ax.scatter(X, Y, Z, c=data[f"{feature}"], marker='o') 
        plt.title(f"{feature}")
        plt.colorbar(p) 
        
    plt.show()
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png


@app.route('/')
def plot_data_age():
    result = get_plot_data('Age', list_data)
    return render_template('plot_age.html', result=result.decode('utf8'))

@app.route('/fare')
def plot_data_fare():
    result = get_plot_data('Fare', list_data)
    return render_template('plot_fare.html', result=result.decode('utf8'))

list_titanic = []
@app.route('/table')
def home():
    for i in iter_file:
        list_titanic.append(i.strip().split(','))
    return render_template('table.html', for_i = list_titanic)



@app.route('/pca')
def plot_data_pca():
    result = get_plot_pca(df)
    return render_template('plot_pca.html', result=result.decode('utf8'))


