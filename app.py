import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

app = Flask(__name__)

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and clustering
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    algorithm = request.form.get('algorithm')  # Get the selected algorithm

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read the CSV file
        data = pd.read_csv(filepath)
        
        # Perform clustering based on the selected algorithm
        if algorithm == 'kmeans':
            clusters = perform_kmeans_clustering(data)
        elif algorithm == 'em':
            clusters = perform_em_clustering(data)

        # Generate and save the plot
        plot_clusters(data, clusters)

        # Redirect to the page showing the plot
        return redirect(url_for('show_plot'))

    return redirect(request.url)

# Function to perform K-Means clustering
def perform_kmeans_clustering(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data.iloc[:, 1:])  # Assuming features start from second column
    return kmeans.labels_

# Function to perform EM (Gaussian Mixture Model) clustering
def perform_em_clustering(data, n_clusters=3):
    gmm = GaussianMixture(n_components=n_clusters)
    gmm.fit(data.iloc[:, 1:])  # Assuming features start from second column
    return gmm.predict(data.iloc[:, 1:])

# Function to plot clusters
def plot_clusters(data, labels):
    x = data.iloc[:, 0]  # Assuming the first column is numeric (for x-axis)
    y = data.iloc[:, 1]  # Assuming the second column is numeric (for y-axis)

    plt.figure(figsize=(10, 6))
    
    # Scatter plot with different colors for different clusters
    scatter = plt.scatter(x, y, c=labels, cmap='viridis', s=100, alpha=0.7, edgecolors='w')
    
    # Add legend
    plt.legend(*scatter.legend_elements(), title="Clusters", loc="best")
    
    # Add labels and title
    plt.xlabel('X-axis (Numeric Value)')
    plt.ylabel('Y-axis (Numeric Value)')
    plt.title('Clustering Results')

    # Add note in the corner
    plt.text(0.95, 0.05, 'Clusters represent different groups', 
             horizontalalignment='right', verticalalignment='bottom',
             transform=plt.gca().transAxes, fontsize=12, color='black',
             bbox=dict(facecolor='white', alpha=0.5))

    # Save plot to the 'static' folder
    plt.savefig('static/cluster_plot.png')  
    plt.close()  # Close the plot to avoid memory issues

# Route to show the plot
@app.route('/plot')
def show_plot():
    return render_template('plot.html')

if __name__ == '__main__':
    app.run(debug=True)
