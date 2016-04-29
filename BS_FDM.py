from numpy.core import empty, clip, zeros, exp, sqrt, ceil
from numpy import linspace
import scipy.sparse.linalg.dsolve as linsolve
from scipy import sparse

class BlackScholesSolver:
  """Finite-difference solver for the Black-Scholes PDE in its most basic form.
     
     The problem to solve is given by:

      Function 
        f = f(t,x) over the rectangle 0 <= t <= T, Bl <= x <= Bu.
      PDE
        rf = df/dt + rx df/dx + (1/2)(sigma x)^2 d^f/dx^2
      Boundary conditions
        given on the three sides of the rectangle
        t = T; x = Bl; x = Bu

     where r, sigma, T, Bl, Bu are given parameters.
  """

  def __init__(self, r, sigma, T, Bl, Bu, Fl, Fu, Fp, m, n):
    """Initialize the finite-difference solver.

     Parameters:
      r     - interest rate
      sigma - volatility
      T     - maturity time
      Bl    - stock price on lower boundary
      Bu    - stock price on upper boundary
      Fl    - value of option on lower boundary
      Fu    - value of option on upper boundary
      Fp    - pay-off at maturity
      m     - number of time steps to take when discretizing PDE
      n     - number of points in x (stock price) domain
              when discretizing PDE;  does not include the boundary points
    """

    self.r  = r;  self.sigma = sigma;  self.T  = T

    self.Bl = Bl;        self.Bu = Bu
    self.Fl = Fl;        self.Fu = Fu

    self.m  = m;         self.n  = n
    
    # Step sizes
    self.dt = float(T)/m
    self.dx = float(Bu-Bl)/(n+1)
    self.xs = Bl/self.dx

    # Grid that will eventually contain the finite-difference solution
    self.u = empty((m+1, n))
    self.u[0,:] = Fp                # initial condition

  def build_sparse_explicit(s):
    """(internal) Set up the sparse matrix system for the explicit method."""

    A = sparse.lil_matrix((s.n, s.n))

    for j in xrange(0, s.n):
      xd = j+1+s.xs
      ssxx = (s.sigma * xd) ** 2
      
      A[j,j] = 1.0 - s.dt*(ssxx + s.r)

      if j > 0:
        A[j,j-1] = 0.5*s.dt*(ssxx - s.r*xd)
      if j < s.n-1:
        A[j,j+1] = 0.5*s.dt*(ssxx + s.r*xd) 

    s.A = A.tocsr()

  def build_sparse_implicit(s):
    """(internal) Set up the sparse matrix system for the implicit method."""

    C = sparse.lil_matrix((s.n, s.n))

    for j in xrange(0, s.n):
      xd = j+1+s.xs
      ssxx = (s.sigma * xd) ** 2
      
      C[j,j] = 1.0 + s.dt*(ssxx + s.r)

      if j > 0:
        C[j,j-1] = 0.5*s.dt*(-ssxx + s.r*xd)
      if j < s.n-1:
        C[j,j+1] = 0.5*s.dt*(-ssxx - s.r*xd) 

    # Store matrix with sparse LU decomposition already performed
    s.C = linsolve.splu(C)

    # Buffer to store right-hand side of the linear system Cu = v
    s.v = empty((n, ))

  def build_sparse_crank_nicolson(s):
    """(internal) Set up the sparse matrices for the Crank-Nicolson method. """

    A = sparse.lil_matrix((s.n, s.n))
    C = sparse.lil_matrix((s.n, s.n))

    for j in xrange(0, s.n):
      xd = j+1+s.xs
      ssxx = (s.sigma * xd) ** 2

      A[j,j] = 1.0 - 0.5*s.dt*(ssxx + s.r)
      C[j,j] = 1.0 + 0.5*s.dt*(ssxx + s.r)
      
      if j > 0:
        A[j,j-1] = 0.25*s.dt*(+ssxx - s.r*xd)
        C[j,j-1] = 0.25*s.dt*(-ssxx + s.r*xd)
      if j < s.n-1:
        A[j,j+1] = 0.25*s.dt*(+ssxx + s.r*xd)
        C[j,j+1] = 0.25*s.dt*(-ssxx - s.r*xd)

    s.A = A.tocsr()
    s.C = linsolve.splu(C)              # perform sparse LU decomposition

    # Buffer to store right-hand side of the linear system Cu = v
    s.v = empty((n, ))
    
  def time_step_explicit(s, i):
    """(internal) Solve the PDE for one time step using the explicit method."""

    # Perform the next time step
    s.u[i+1,:]      = s.A * s.u[i,:]
    
    # and mix in the two other boundary conditions not accounted for above
    xdl = 1+s.xs;  xdu = s.n+s.xs
    s.u[i+1,0]     += s.Fl[i] * 0.5*s.dt*((s.sigma*xdl)**2 - s.r*xdl)
    s.u[i+1,s.n-1] += s.Fu[i] * 0.5*s.dt*((s.sigma*xdu)**2 + s.r*xdu)

  def time_step_implicit(s, i):
    """(internal) Solve the PDE for one time step using the implicit method."""

    s.v[:]      = s.u[i,:]

    # Add in the two other boundary conditions
    xdl = 1+s.xs;  xdu = s.n+s.xs
    s.v[0]     -= s.Fl[i+1] * 0.5*s.dt*(-(s.sigma*xdl)**2 + s.r*xdl)
    s.v[s.n-1] -= s.Fu[i+1] * 0.5*s.dt*(-(s.sigma*xdu)**2 - s.r*xdu)

    # Perform the next time step
    s.u[i+1,:] = s.C.solve(s.v)

  def time_step_crank_nicolson(s, i):
    """(internal) Solve the PDE for one time step using the Crank-Nicolson method."""

    # Perform explicit part of time step
    s.v[:]      = s.A * s.u[i,:]

    # Add in the two other boundary conditions
    xdl = 1+s.xs;  xdu = s.n+s.xs
    s.v[0]     += s.Fl[i]   * 0.25*s.dt*(+(s.sigma*xdl)**2 - s.r*xdl)
    s.v[s.n-1] += s.Fu[i]   * 0.25*s.dt*(+(s.sigma*xdu)**2 + s.r*xdu)
    s.v[0]     -= s.Fl[i+1] * 0.25*s.dt*(-(s.sigma*xdl)**2 + s.r*xdl)
    s.v[s.n-1] -= s.Fu[i+1] * 0.25*s.dt*(-(s.sigma*xdu)**2 - s.r*xdu)
    
    # Perform implicit part of time step
    s.u[i+1,:] = s.C.solve(s.v)

  def solve(self, method='crank-nicolson'):
    """Solve the Black-Scholes PDE with the parameters given at initialization.

      Arguments:
       method - Indicates which finite-difference method to use.  Choices:
                'explicit': explicit method
                'implicit': implicit method
                'crank-nicolson': Crank-Nicolson method
                'smoothed-crank-nicolson': 
                   Crank-Nicolson method with initial smoothing 
                   by the implicit method
    """

    i_start = 0

    if method == 'implicit':
      self.build_sparse_implicit()
      time_step = self.time_step_implicit
    elif method == 'explicit':
      self.build_sparse_explicit()
      time_step = self.time_step_explicit
    elif method == 'crank-nicolson' or method is None:
      self.build_sparse_crank_nicolson()
      time_step = self.time_step_crank_nicolson
    elif method == 'smoothed-crank-nicolson':
      self.build_sparse_implicit()
      for i in range(0, 10):
        self.time_step_implicit(i)
      i_start = 10
      self.build_sparse_crank_nicolson()
      time_step = self.time_step_crank_nicolson
    else:
      raise ValueError('incorrect value for method argument')

    for i in xrange(i_start, m):
      time_step(i)

    return self.u

def european_call(r, sigma, T, Bu, m, n, Bl=0.0, barrier=None, method=None):
  """Compute prices for a European-style call option."""

  X = linspace(0.0, B, n+2)
  X = X[1:-1]

  Fp = clip(X-K, 0.0, 1e600)
  
  if barrier is None:
    Fu = B - K*exp(-r * linspace(0.0, T, m+1))
    Fl = zeros((m+1, ))
  elif barrier == 'up-and-out':
    Fu = Fl = zeros((m+1,))

  bss = BlackScholesSolver(r, sigma, T, Bl, Bu, Fl, Fu, Fp, m, n)
  return X, bss.solve(method)

def european_put(r, sigma, T, Bu, m, n, Bl=0.0, barrier=None, method=None):
  """Compute prices for a European-style put option."""

  X = linspace(0.0, B, n+2)
  X = X[1:-1]

  Fp = clip(K-X, 0.0, 1e600)
  
  if barrier is None:
    Fu = zeros((m+1,))
    Fl = K*exp(-r * linspace(0.0, T, m+1))
  elif barrier == 'up-and-out':
    Fu = Fl = zeros((m+1,))

  bss = BlackScholesSolver(r, sigma, T, Bl, Bu, Fl, Fu, Fp, m, n)
  return X, bss.solve(method)

def plot_solution(T, X, u):
  
  # The surface plot
  '''
  Xm, Tm = pylab.meshgrid(X, linspace(T, 0.0, u.shape[0]))
  fig_surface = pylab.figure()
  # ax = matplotlib.axes3d.Axes3D(fig_surface)
  ax = Axes3D(fig_surface)
  ax.plot_surface(Xm, Tm, u)
  ax.set_ylabel('Time $t$')
  ax.set_xlabel('Stock price $x$')
  ax.set_zlabel('Option value $f(t,x)$')
  '''
  # The color temperature plot
  fig_color = pylab.figure()
  ax = pylab.gca()
  ax.set_xlabel('Time $t$')
  ax.set_ylabel('Stock price $x$')
  ax.imshow(u.T, interpolation='bilinear', origin='lower', 
            cmap=matplotlib.cm.hot, aspect='auto', extent=(T,0.0, X[0],X[-1]))
  
  # Plot of price function at time 0
  fig_zero = pylab.figure()
  pylab.plot(X, u[m-1,:])
  ax = pylab.gca()
  ax.set_xlabel('Stock price $x$')
  ax.set_ylabel('Option value $f(0,x)$')

  return fig_color, fig_zero

def parse_options():
  from optparse import OptionParser

  parser = OptionParser()

  parser.add_option("-r", "--interest", dest="r", default="0.10",
                    help="interest rate [default: %default]")
  parser.add_option("-v", "--volatility", dest="sigma", default="0.40",
                    help="volatility [default: %default]")
  
  parser.add_option("-K", "--strike", dest="K", default="50.00",
                    help="strike price [default: %default]")
  parser.add_option("-T", "--maturity", dest="T", default="0.5",
                    help="maturity time [default: %default]")
  parser.add_option("-B", "--bound", dest="B", default="100.00",
                    help="upper bound on stock price [default: %default]")

  parser.add_option("-m", "--time-steps", dest="m", default="100",
                    help="number of time steps [default: %default]")
  parser.add_option("-n", "--space-steps", dest="n", default="200",
                    help="number of steps in stock-price space [default: %default]")
  parser.add_option("--dt", dest="dt", help="time step size")
  parser.add_option("--dx", dest="dx", help="stock-price step size")
  parser.add_option("--method", dest="method", help="finite-difference method")

  parser.add_option("-C", "--call", dest="is_call", action="store_true", 
                    help="value a European-style call option")
  parser.add_option("-P", "--put", dest="is_put", action="store_true", 
                    help="value a European-style put option")
  parser.add_option("--barrier", dest="barrier",
                    help="value a barrier option")

  parser.add_option("--plot", dest="plot", action="store_true", 
                    help="plot results")
  parser.add_option("--save-plot", dest="save_plot", action="store_true",
                    help="save plots to EPS files")

  (options, args) = parser.parse_args()
  return options

if __name__ == "__main__":
  options = parse_options()

  # Parameters
  r = float(options.r)
  sigma = float(options.sigma)
  K = float(options.K)
  T = float(options.T)
  B = float(options.B)
  
  m = int(options.m)
  n = int(options.n)

  if options.dt is not None:
    m = ceil(T/float(options.dt))
  if options.dx is not None:
    n = ceil(B/float(options.dx)) - 1

  if options.is_put:
    X, u = european_put(r, sigma, T, B, m, n, 
                        barrier=options.barrier, 
                        method=options.method)
  else:
    X, u = european_call(r, sigma, T, B, m, n, 
                        barrier=options.barrier,
                        method=options.method)
  
  # Print out results at time 0
  print "Stock price x    Option price f(0,x)"
  for i in xrange(0, n):
    print "%10.4f         %10.4f " % (X[i], u[m,i])

  # Generate plots if user requests
  if options.plot or options.save_plot:
    import pylab
    import matplotlib
    from mpl_toolkits.mplot3d import Axes3D

    golden_mean = (sqrt(5.0)-1.0)/2.0
    pylab.rcParams.update( \
      {'backend': 'ps',
       'ps.usedistiller': 'xpdf',
       'axes.labelsize': 10,
       'text.fontsize': 10,
       'xtick.labelsize': 8,
       'ytick.labelsize': 8,
       'figure.figsize': [ 7.0, golden_mean*7.0 ],
       'text.usetex': True })

    fig_color, fig_zero = plot_solution(T, X, u)
  
    # Show figures
    if options.plot:
      pylab.show()

    # Save figures to EPS format
    if options.save_plot:
      # fig_surface.savefig('bs-surface.eps')
      #fig_color.savefig('bs-color.eps')
      fig_zero.savefig('bs-zero.eps')