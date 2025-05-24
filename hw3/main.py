from part1 import SampleData, CrossValidation, SimpleMLP, SVM
from pprint import pprint as pp

# create function to make this code less repetitive and more readable

def compare_models(n_samples, k, model, data_name, model_name):

  print(f'Generating {data_name}')
  sampler = SampleData(n_samples=n_samples, data_name=data_name)
  data = sampler.generate_data()
  print(f'{data_name} generated')

  print(f'Plotting {data_name}')
  #sampler.plot(data[0], data[1]) # debug why console shows plot in VS Code but gets stuck 
  print(f'{data_name} plotted')

  print(f'Splitting {data_name} into {k} folds')
  cv = CrossValidation(data=data, k=k, model=model, data_name=data_name, model_name=model_name)
  cv.split_data()
  print(f'{data_name} split into {k} folds')

  print(f'Training {model_name} on {data_name}')
  results = cv.train_test()
  print(f'Results for {data_name} with {model_name}')
  pp(results)

def main():

  # cross validation for Simple MLP and SVM with each kernel: Linear, Polynomial, RBF, and Sigmoid
  k = 5
  n = 10 # number of hidden units
  datasets = {
    'Half Circles (S)': 1000, 
    'Half Circles (L)': 10000, 
    'Half Moons (S)': 1000,
    'Half Moons (L)': 10000
  }

  models = {
    'MLP': SimpleMLP(size_layers=[2, n, 1]), 
    'SVM-Linear': SVM(kernel='linear'), 
    'SVM-Poly': SVM(kernel='poly'), 
    'SVM-RBF': SVM(kernel='rbf'), 
    'SVM-Sigmoid': SVM(kernel='sigmoid')
  }

  for dname, size in datasets.items():
    for mname, model in models.items():
      compare_models(size, k, model, dname, mname)


if __name__ == "__main__":
  try:
    main()
  except Exception as e:
    print(f"Error during execution: {e}")
