use std::cell::RefCell;
use std::ops::{Add, Mul, Sub};
use std::rc::Rc;

#[derive(Clone, Debug, PartialEq)]
enum Op {
    None,
    Add,
    Sub,
    Mul,
}

type Children = (Rc<RefCell<Value>>, Rc<RefCell<Value>>);

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

    pub fn children(&self) -> (Rc<RefCell<Value>>, Rc<RefCell<Value>>) {
        self.children.clone().unwrap()
    }

    pub fn backprop(&mut self) {
        self.grad = 1.;
        self.backprop_internal();
    }

    fn backprop_internal(&mut self) {
        if let Some((ref a, ref b)) = self.children {
            match self.op {
                Op::Add => {
                    a.borrow_mut().grad += self.grad;
                    b.borrow_mut().grad += self.grad;
                }
                Op::Sub => {
                    a.borrow_mut().grad += self.grad;
                    b.borrow_mut().grad -= self.grad;
                }
                Op::Mul => {
                    a.borrow_mut().grad += self.grad * b.borrow().data();
                    b.borrow_mut().grad += self.grad * a.borrow().data();
                }
                _ => {}
            }
            a.borrow_mut().backprop_internal();
            b.borrow_mut().backprop_internal();
        }
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, other: Value) -> Self {
        Self::new_with_children(
            self.data + other.data,
            (Rc::new(RefCell::new(self)), Rc::new(RefCell::new(other))),
            Op::Add,
        )
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, other: Value) -> Self {
        Self::new_with_children(
            self.data - other.data,
            (Rc::new(RefCell::new(self)), Rc::new(RefCell::new(other))),
            Op::Sub,
        )
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, other: Value) -> Self {
        Self::new_with_children(
            self.data * other.data,
            (Rc::new(RefCell::new(self)), Rc::new(RefCell::new(other))),
            Op::Mul,
        )
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
        let mut d = c * Value::new(3.);
        d.backprop();
        let (c, e) = d.children();
        assert_eq!(c.borrow().grad, 3.0);
        assert_eq!(e.borrow().grad, -2.0);
        dbg!(d);
    }
}
