# Ising Examples

## Saving Models
One can save a set of models as determined in the ```params.py``` file, as well as projections onto symmetry sectors, by running ```save_models.py```.
The models are saved as dictionaries in the ```Ising/operators``` folder. The code for this procedure, and the keys of the dictionaries, can be found in ```Ising.ising.utils.calculation_utils.eigh_symms```.

## Thermalization Basics: Thermalization of Small Operators

### Disclaimer
Based almost entirely on code given to me by the extremely generous and knowledgeable Nick O'Dea. It is my understanding that he has developed this code in part in collaboration with Vedika Khemani. I am extremely grateful to both of them!

The particular choices of parameters in this code are motivated by 1308.2862.pdf, which states that these parameters lead to demonstrable non-integrable behavior even for small system sizes.
Since this code is run locally, this is a powerful asset that reduces the computational power necessary to observe non-integrability.

### Mixed Field Ising Model
The Mixed Field Ising model is a remarkably simple lattice model which nonetheless exhibits non-integrable behavior. It is a simple, nearest neighbor Ising model with both transverse and longitudinal fields.
When the longitudinal field is turned off, the model is integrable. As it turns on, we see non-integrable behavior, including thermalization, as
demonstrated by the code of Khemani/O'Dea.

### XXZ Model with Next-to-Nearest-Neighbor Interactions
The XXZ model is an Ising-type model with nearest neighbor quadratic X, Y, and Z interactions, where the X and Y interactions have the same strength and the Z interaction does not.

When we add in next-to-nearest neighbor interactions, this is another model which can be used to study non-integrable behavior.

### Resources
Finite-size scaling of eigenstate thermalization:
https://arxiv.org/pdf/1308.2862.pdf

Ballistic spreading of entanglement in a diffusive nonintegrable system:
https://arxiv.org/pdf/1306.4306.pdf


## Thermalization and Pseudorandomness: Probing Thermalization of Large Operators
