# https://www.desmos.com/calculator/gi6q3nvlnw

class SDF:
    
    def __init__(self):
        self.pts = []
        self.a = 0.05
        self.b = 0.25
        self.c = 1
    
    def add(self, x, y):
        self.pts.append((x, y))
        
    def grad(self, x, y):
        
        grad_x, grad_y = 0, 0
        for xi, yi in self.pts:
            
            diff_x = x - xi
            diff_y = y - yi
            dist = (diff_x**2 + diff_y**2)**0.5
            
            if dist > self.b: grad = 0
            else:
                if dist < self.a: grad = self.a ** (-3)
                else: grad = dist ** (-3)
                grad *= -2
                
            grad_x += grad * diff_x / dist
            grad_y += grad * diff_y / dist
        
        grad_x *= self.c
        grad_y *= self.c
        return grad_x, grad_y