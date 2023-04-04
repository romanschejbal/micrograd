use super::Value;

pub struct Neuron {
    w: Vec<Value>,
    b: Value,
}

impl Neuron {
    pub fn new(nin: usize) -> Self {
        let random_number = || rand::random::<f32>() * 2.0 - 1.0;
        Self {
            w: (0..nin)
                .into_iter()
                .map(|_| Value::new(random_number()))
                .collect(),
            b: Value::new(random_number()),
        }
    }

    pub fn forward(&mut self, x: &[f32]) -> Value {
        let sum = &mut self.b;
        for (w, x) in self.w.iter_mut().zip(x.iter()) {
            sum = sum + w * x;
        }
        let out = sum.tanh();
        out
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Self {
        Self {
            neurons: (0..nout).into_iter().map(|_| Neuron::new(nin)).collect(),
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(x))
            .collect()
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, nouts: &[usize]) -> Self {
        let mut layers = vec![];
        let mut nin = nin;
        for nout in nouts {
            layers.push(Layer::new(nin, *nout));
            nin = *nout;
        }
        Self { layers }
    }

    // pub fn forward(&self, x: &[f32]) -> Vec<Value> {
    //     let mut x = x.to_vec();
    //     for layer in self.layers.iter() {
    //         x = layer.forward(&x).iter().map(|v| v.data()).collect();
    //     }
    //     x
    // }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward() {
        let neuron = Neuron::new(2);
        let x = [1.0, 2.0];
        let y = neuron.forward(&x);
        assert_eq!(y.data(), 1.5);
    }
}
