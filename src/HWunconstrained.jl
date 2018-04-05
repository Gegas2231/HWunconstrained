

module HWunconstrained

	# imports: which packages are we going to use in this module?
	using Distributions, Optim, Plots, DataFrames

	"""
    `input(prompt::AbstractString="")`

    Read a string from STDIN. The trailing newline is stripped.

    The prompt string, if given, is printed to standard output without a
    trailing newline before reading input.
    """
    function input(prompt::AbstractString="")
        print(prompt)
        return chomp(readline())
    end

    export maximize_like_grad, runAll, makeData



	# methods/functions
	# -----------------

	# data creator
	# should/could return a dict with beta,numobs,X,y,norm)
	# true coeff vector, number of obs, data matrix X, response vector y, and a type of parametric distribution for G.

	function makeData(n=10_000)
		beta = [ 1; 1.5; -0.5 ]
		#No need for more complex functions since var/covar matrix show no joint distribution
		X = randn((n,size(beta,1)))
		#Using an explicit form for Y, and since we say is a probit model, we compute an normally distributed (0,1) error term.
		ϵ = randn(n)
		#Since the error is centered, we have that y = 1 if -epsilon > Xbeta or Y > 0
		y = X*beta + ϵ
		y = 1*(y.>0)
		norm =  Normal()
		return Dict("beta"=>beta,"numobs"=>n,"X"=>X,"y"=>y,"dist"=>norm)
	end

	# log likelihood function at x
	# function loglik(betas::Vector,X::Matrix,y::Vector,distrib::UnivariateDistribution)
	function loglik(β::Vector,d::Dict)
		# β = d["beta"]
		X = d["X"]
		y = d["y"]
		n = d["numobs"]
		dist = d["dist"]
		Xβ = X * β
		objective = 0.0
		for i in 1:n
			objective = objective + y[i] * log( cdf(dist, Xβ[i]) ) + (1-y[i]) * log( 1 - cdf(dist, Xβ[i]) )
		end
		return objective
	end

	# gradient of the likelihood at x
	function grad!(storage::Vector,betas::Vector,d)
		xbeta     = d["X"]*betas	# (n,1)
		G_xbeta   = cdf.(d["dist"],xbeta)	# (n,1)
		g_xbeta   = pdf.(d["dist"],xbeta)	# (n,1)
		storage = -mean((d["y"] .* g_xbeta ./ G_xbeta - (1-d["y"]) .* g_xbeta ./ (1-G_xbeta)) .* d["X"],1)
		return nothing
	end

	# hessian of the likelihood at x
	function hessian!(storage::Matrix,betas::Vector,d)
		return nothing
	end


	function info_mat(betas::Vector,d)
	end

	function inv_Info(betas::Vector,d)
	end



	"""
	inverse of observed information matrix
	"""
	function inv_observedInfo(betas::Vector,d)
	end

	"""
	standard errors
	"""
	function se(betas::Vector,d::Dict)
	end

	# function that maximizes the log likelihood without the gradient
	# with a call to `optimize` and returns the result
	function maximize_like(d::Dict,x0=[0.8,1.0,-0.1],meth=NelderMead())
		β_hat = optimize(arg->-loglik(arg,d),x0,meth, Optim.Options(iterations = 500,g_tol=1e-20))
		return β_hat
	end

	function maximize_like_helpNM(d::Dict,x0=[ 1; 1.5; -0.5 ],meth=NelderMead())
		β_hat = optimize(arg->loglik(arg,d),x0,meth, Optim.Options(iterations = 500,g_tol=1e-20))
		return β_hat
	end



	# function that maximizes the log likelihood with the gradient
	# with a call to `optimize` and returns the result
	function maximize_like_grad(d::Dict,x0=[0.8,1.0,-0.1],meth=BFGS())
		res = optimize(arg->-loglik(arg,d), grad!, x0,meth, Optim.Options(iterations = 500,g_tol=1e-20))
		return res
	end

	function maximize_like_grad_hess(x0=[0.8,1.0,-0.1],meth=Newton())
		return res
	end

	function maximize_like_grad_se(x0=[0.8,1.0,-0.1],meth=BFGS())
	end


	# visual diagnostics
	# ------------------

	# function that plots the likelihood
	# we are looking for a figure with 3 subplots, where each subplot
	# varies one of the parameters, holding the others fixed at the true value
	# we want to see whether there is a global minimum of the likelihood at the the true value.
	function plotLike(d::Dict)
		beta = [ 1; 1.5; -0.5 ]
		points_β1 = collect(beta[1]-1:0.01:beta[1]+1)
		points_β2 = collect(beta[2]-1:0.01:beta[2]+1)
		points_β3 = collect(beta[3]-1:0.01:beta[3]+1)
		loglik_β1 = [ loglik( [ b, beta[2], beta[3] ], d ) for b in points_β1]
		loglik_β2 = [ loglik( [ beta[1], b ,beta[3] ], d ) for b in points_β2]
		loglik_β3 = [ loglik( [ beta[1], beta[2] ,b ], d ) for b in points_β3]
		max_loglik = loglik(beta,d)
		plot_β1 = plot(points_β1, loglik_β1, labels = "Beta1", legend=:bottomright)
		scatter!(plot_β1, [beta[1]], [max_loglik], labels = "Maximizer")
		plot_β2 = plot(points_β2, loglik_β2, labels = "Beta2", legend=:bottomright)
		scatter!(plot_β2, [beta[2]], [max_loglik], labels = "Maximizer")
		plot_β3 = plot(points_β3, loglik_β3, labels = "Beta3", legend=:bottomleft)
		scatter!(plot_β3, [beta[3]], [max_loglik], labels = "Maximizer")
		plotlik = plot(plot_β1, plot_β2, plot_β3, layout=(1,3))
	end

	function plotGrad()
		d = makeData(10000)
	end



	function runAll()
		srand(1)
		d = makeData()
		plotLike(d)
		#plotGrad()
		m1 = maximize_like(d, [0.8,1.0,-0.1], NelderMead())
		#m2 = maximize_like_grad()
		#m3 = maximize_like_grad_hess()
		#m4 = maximize_like_grad_se()
		println("results are:")
		println("maximize_like optimizer: $(m1.minimizer)")
		println("maximize_like iterations: $(Optim.iterations(m1))")
		#println("maximize_like_grad: $(Optim.minimizer(m2))")
		#println("maximize_like_grad iterations: $(Optim.iterations(m2))")
		#println("maximize_like_grad_hess: $(m3.minimizer)")
		#println("maximize_like_grad_hess iterations: $(Optim.iterations(m3))")
		#println("maximize_like_grad_se: $m4)")
		println("")
		println("running tests:")
		include("test/runtests.jl")
		println("")
		if isinteractive()
			ok = input("enter y to close this session.")
			if ok == "y"
				quit()
			end
		end
	end


end
