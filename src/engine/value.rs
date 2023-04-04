use std::{
    cell::RefCell,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

#[derive(Clone, Debug, PartialEq)]
enum Op {
    None,
    Add,
    Mul,
    Tanh,
    Exp,
    Pow,
}

type SharedValue = Rc<RefCell<Value>>;
type Children = (Option<SharedValue>, Option<SharedValue>);

#[derive(Clone, Debug, PartialEq)]
pub struct Value {
    data: f32,
    op: Op,
    children: Option<Children>,
    grad: f32,
}

impl Value {
    pub fn new(data: f32) -> Self {
        Self {
            data,
            op: Op::None,
            children: None,
            grad: 0.,
        }
    }

    fn new_with_children(data: f32, children: Children, op: Op) -> Self {
        Self {
            data,
            op,
            children: Some(children),
            grad: 0.,
        }
    }

    pub fn data(&self) -> f32 {
        self.data
    }

    pub fn backprop(&mut self) {
        // let mut topo = vec![];
        // fn build_topo(topo: &mut Vec<Rc<RefCell<Value>>>, parent: Rc<RefCell<Value>>) {
        //     if let Some((ref a, ref b)) = parent.borrow().children {
        //         a.as_ref()
        //             .map(|children| build_topo(topo, Rc::clone(children)));
        //         b.as_ref()
        //             .map(|children| build_topo(topo, Rc::clone(children)));
        //     }
        //     topo.push(Rc::clone(&parent));
        // }

        // if let Some((ref a, ref b)) = self.children {
        //     a.as_ref()
        //         .map(|children| build_topo(&mut topo, Rc::clone(children)));
        //     b.as_ref()
        //         .map(|children| build_topo(&mut topo, Rc::clone(children)));
        // }

        self.grad = 1.;
        self.backprop_internal();
        // topo.iter()
        //     .rev()
        //     .for_each(|v| v.borrow_mut().backprop_internal());
    }

    fn backprop_internal(&mut self) {
        println!("Backprop! {:?} {:?}", &self.op, self.data);
        if let Some((Some(ref a), Some(ref b))) = self.children {
            match self.op {
                Op::None => {}
                Op::Add => {
                    a.borrow_mut().grad += self.grad;
                    b.borrow_mut().grad += self.grad;
                }
                Op::Mul => {
                    a.borrow_mut().grad += self.grad * b.borrow().data;
                    b.borrow_mut().grad += self.grad * a.borrow().data;
                }
                Op::Pow => {
                    a.borrow_mut().grad +=
                        b.borrow().data * (a.borrow().data.powf(b.borrow().data - 1.)) * self.grad;
                }
                _ => unreachable!(),
            }
            a.borrow_mut().backprop_internal();
            b.borrow_mut().backprop_internal();
        } else if let Some((Some(ref a), None)) = self.children {
            match self.op {
                Op::Tanh => {
                    a.borrow_mut().grad += (1. - self.data.powf(2.)) * self.grad;
                }
                Op::Exp => {
                    a.borrow_mut().grad += self.data * self.grad;
                }
                _ => unreachable!(),
            }
            a.borrow_mut().backprop_internal();
        }
    }

    pub fn tanh(self) -> Self {
        let x = self.data;
        let t = (f32::EPSILON.powf(2. * x) - 1.) / (f32::EPSILON.powf(2. * x) + 1.);
        Self::new_with_children(t, (Some(Rc::new(RefCell::new(self))), None), Op::Tanh)
    }

    pub fn exp(self) -> Self {
        let x = self.data;
        Self::new_with_children(
            f32::EPSILON.powf(x),
            (Some(Rc::new(RefCell::new(self))), None),
            Op::Exp,
        )
    }

    pub fn pow(self, exp: f32) -> Self {
        Self::new_with_children(
            self.data.powf(exp),
            (
                Some(Rc::new(RefCell::new(self))),
                Some(Rc::new(RefCell::new(Value::new(exp)))),
            ),
            Op::Pow,
        )
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, other: Value) -> Self {
        Self::new_with_children(
            self.data + other.data,
            (
                Some(Rc::new(RefCell::new(self))),
                Some(Rc::new(RefCell::new(other))),
            ),
            Op::Add,
        )
    }
}

impl Add<f32> for Value {
    type Output = Value;

    fn add(self, other: f32) -> Self {
        self.add(Value::new(other))
    }
}

impl Add<Value> for f32 {
    type Output = Value;

    fn add(self, other: Value) -> Value {
        Value::new(self).add(other)
    }
}

impl Add<Value> for &mut Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        self.add(rhs)
    }
}

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self {
        self * -1.
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, other: Value) -> Self {
        Self::add(self, -other)
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, other: Value) -> Self {
        Self::new_with_children(
            self.data * other.data,
            (
                Some(Rc::new(RefCell::new(self))),
                Some(Rc::new(RefCell::new(other))),
            ),
            Op::Mul,
        )
    }
}

impl Mul<f32> for Value {
    type Output = Value;

    fn mul(self, other: f32) -> Self {
        self.mul(Value::new(other))
    }
}

impl Mul<Value> for f32 {
    type Output = Value;

    fn mul(self, other: Value) -> Value {
        Value::new(self).mul(other)
    }
}

impl Mul<&f32> for &mut Value {
    type Output = Value;

    fn mul(self, rhs: &f32) -> Self::Output {
        self.mul(rhs)
    }
}

impl Div for Value {
    type Output = Value;

    fn div(self, other: Self) -> Self::Output {
        self * other.pow(-1.)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        let a = Value::new(-4.0);
        let b = Value::new(2.0);
        let c = a + b;
        assert_eq!(c.data(), -2.0);
    }

    #[test]
    fn sub() {
        let a = Value::new(-4.0);
        let b = Value::new(2.0);
        let c = a - b;
        assert_eq!(c.data(), -6.0);
    }

    #[test]
    fn mul() {
        let a = Value::new(-4.0);
        let b = Value::new(2.0);
        let c = a * b;
        assert_eq!(c.data(), -8.0);
    }

    #[test]
    fn backprop() {
        let a = Value::new(-4.0);
        let b = Value::new(2.0);
        let c = a + b;
        let d = c * Value::new(1.);
        // let e = d * Value::new(1.);

        let mut o = d.tanh() * Value::new(5.);
        o.backprop();

        dbg!(o);
    }
}
