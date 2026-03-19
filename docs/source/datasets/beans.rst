Beans
=====

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Plant%20Disease%20Classification-blue" alt="Task: Plant Disease Classification">
   <img src="https://img.shields.io/badge/Classes-3-green" alt="Classes: 3">
   <img src="https://img.shields.io/badge/Domain-Agriculture-orange" alt="Domain: Agriculture">
   </p>

Overview
--------

The Beans dataset (also known as IBeans) contains leaf images for classification of bean plant diseases. The dataset represents three classes: healthy leaves, angular leaf spot disease, and bean rust disease. Images were collected in Uganda for practical disease classification in the field.

- **Train**: 1,034 images
- **Validation**: 133 images
- **Test**: 128 images

This dataset is particularly valuable for developing agricultural AI applications to help farmers identify and manage crop diseases early.

Data Structure
--------------

When accessing an example using ``ds[i]``, you will receive a dictionary with the following keys:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Type
     - Description
   * - ``image``
     - ``PIL.Image.Image``
     - RGB image of bean leaf (variable dimensions, typically ~500x500)
   * - ``label``
     - int
     - Disease class label (0-2)

Disease Classes
---------------

The dataset includes 3 disease classes:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Label
     - Class Name
   * - 0
     - **healthy** - Healthy bean leaves with no disease
   * - 1
     - **angular_leaf_spot** - Angular leaf spot disease caused by *Pseudocercospora griseola*
   * - 2
     - **bean_rust** - Bean rust disease caused by *Uromyces appendiculatus*

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.beans import Beans

    # Load different splits
    train_ds = Beans(split="train")
    val_ds = Beans(split="validation")
    test_ds = Beans(split="test")

    # Inspect a sample
    sample = train_ds[0]
    print(sample.keys())  # {"image", "label"}
    print(f"Label: {sample['label']}")  # 0, 1, or 2

    # Optional: make it PyTorch-friendly
    train_torch = train_ds.with_format("torch")

**Getting Class Names**

.. code-block:: python

    from stable_datasets.images.beans import Beans

    ds = Beans(split="train")
    
    # Get class names
    class_names = ds.info.features["label"].names
    print(class_names)  # ["healthy", "angular_leaf_spot", "bean_rust"]
    
    # Map label to class name
    sample = ds[0]
    disease_name = class_names[sample["label"]]
    print(f"Disease: {disease_name}")

**Typical Workflow: Disease Classification**

.. code-block:: python

    from stable_datasets.images.beans import Beans
    from torchvision import transforms
    from torch.utils.data import DataLoader

    # Load and prepare data
    train_ds = Beans(split="train")
    val_ds = Beans(split="validation")
    
    # Convert to PyTorch format
    train_ds = train_ds.with_format("torch")
    val_ds = val_ds.with_format("torch")
    
    # Apply transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_ds.set_transform(lambda x: {
        "image": transform(x["image"]), 
        "label": x["label"]
    })
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    
    # ... train your disease classification model ...

Dataset Statistics
------------------

The dataset is relatively small but balanced across the three classes, making it suitable for:

- Transfer learning and fine-tuning pre-trained models
- Benchmarking few-shot learning algorithms
- Educational purposes in agricultural AI
- Prototyping crop disease detection systems

References
----------

- Homepage: https://github.com/AI-Lab-Makerere/ibean/
- License: MIT License
- Source: Makerere AI Lab, Uganda

Citation
--------

.. code-block:: bibtex

    @misc{makerere2020beans,
      author = "{Makerere AI Lab}",
      title = "{Bean Disease Dataset}",
      year = "2020",
      month = "January",
      url = "https://github.com/AI-Lab-Makerere/ibean/"
    }
