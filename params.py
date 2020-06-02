# Wybrane hiperparametry

channels = 3
img_width = 128
img_height = 128
img_shape = (channels, img_width, img_height)
residual_blocks = 6

selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
c_dim = len(selected_attrs)

label_changes = [
  ((0, 1), (1, 0), (2, 0)),  # Ustawienie czarnych włosów
  ((0, 0), (1, 1), (2, 0)),  # Ustawienie blond włosów
  ((0, 0), (1, 0), (2, 1)),  # Ustawienie brązowych włosów
  ((3, -1),),  # Zmiana płci
  ((4, -1),),  # Zmiana wieku
]
