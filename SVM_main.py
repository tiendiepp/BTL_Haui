from cell_analysis.data_loading import load_and_prepare_data
from cell_analysis.data_visualization import plot_data
from cell_analysis.model_training import train_svm_model
from cell_analysis.model_evaluation import evaluate_model, plot_confusion_matrix

cell_df = load_and_prepare_data("cell_samples.csv")
plot_data(cell_df)
clf, X_test, y_test = train_svm_model(cell_df)
cnf_matrix = evaluate_model(clf, X_test, y_test)
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)', 'Malignant(4)'], normalize=False, title='Confusion matrix')
