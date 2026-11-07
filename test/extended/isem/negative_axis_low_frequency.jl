# Negative imaginary axis with |omega| <= 0.1 (principal-MST route): the route carried no amplitude errors, so its
# certificate read them as Inf and Y_radial threw "unusable estimated_error=Inf" at every such frequency (Figure 2
# fiducial mode, y = 1.06e-6 to 0.097). The sign of the zero real part selects the side of the cut: -0.0 the limit
# from Re omega < 0, +0.0 from Re omega > 0. References: the two-ray routes 1e-12 |omega| off the axis on either side;
# B_inc moves by about 5e-12 over that offset, Y is continuous across the cut.
@testset "negative imaginary axis, |omega| <= 0.1" begin
    for y in (1.0624612090753808e-6, 1e-3, 0.096969271), side in (-1, 1)
        axis = Y_radial(-2, 2, 2, 0.9, complex(side * 0.0, -y), IN)
        lip = Y_radial(-2, 2, 2, 0.9, complex(side * 1e-12y, -y), IN)
        certified, estimate, _ = GSN.ISEM.certificate(axis)
        @test isfinite(estimate) && estimate < 1e-12
        @test axis.incidence_amplitude ≈ lip.incidence_amplitude rtol=2e-11
        @test axis(5.0) ≈ lip(5.0) rtol=1e-11
    end
end
