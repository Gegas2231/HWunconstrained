


@testset "basics" begin

	@testset "Test Data Construction" begin
		@test size(HWunconstrained.makedata()["X"]) == (10000, 3)
		@test size(HWunconstrained.makedata()["y"]) == (10000, 1)

		#@test any(1, HWunconstrained.makedata()["y"])
	end

	@testset "Test Return value of likelihood" begin
		#Negative loglikelihood function
		@test HWunconstrained.loglik(HWunconstrained.makedata()["beta"], HWunconstrained.makedata()) < 0
		#Loglikelihood is higher at the optimal beta
		@test HWunconstrained.loglik(HWunconstrained.makedata()["beta"], HWunconstrained.makedata()) > HWunconstrained.loglik([1,1,1], HWunconstrained.makedata())
		@test HWunconstrained.loglik(HWunconstrained.makedata()["beta"], HWunconstrained.makedata()) > HWunconstrained.loglik([1.5,1.5,1.5], HWunconstrained.makedata())
		@test HWunconstrained.loglik(HWunconstrained.makedata()["beta"], HWunconstrained.makedata()) > HWunconstrained.loglik([-0.5,-0.5,-0.5], HWunconstrained.makedata())
	end

	#@testset "Test return value of gradient" begin

	#end
end

@testset "test maximization results" begin

	@testset "maximize returns approximate result" begin
		@test HWunconstrained.maximize_like().minimizer[1] ≈ 1 atol = 0.1
		@test HWunconstrained.maximize_like().minimizer[2] ≈ 1.5 atol = 0.1
		@test HWunconstrained.maximize_like().minimizer[3] ≈ -0.5 atol = 0.1
	end

#	@testset "maximize_grad returns accurate result" begin
#	end
#
#	@testset "maximize_grad_hess returns accurate result" begin
#	end
#
	@testset "gradient is close to zero at max like estimate" begin
		gradient = Vector(3)
		HWunconstrained.grad!(gradient, HWunconstrained.maximize_like_grad().minimizer, HWunconstrained.makedata())
		@test gradient[1] ≈ 0 atol = 1.0e-5
		@test gradient[2] ≈ 0 atol = 1.0e-5
		@test gradient[3] ≈ 0 atol = 1.0e-5
	end

end

#@testset "test against GLM" begin

#	@testset "estimates vs GLM" begin


#	end

#	@testset "standard errors vs GLM" begin


#	end

end
