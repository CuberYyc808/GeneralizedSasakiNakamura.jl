# Horizon seed near m Omega_H: (l, m) = (2, 2), a = 0.99, omega = 0.80 - 0.87. With the real-frequency seed capped at
# x = 0.3 it jumped between about 0.065 and 0.3 from one omega to the next, and B_inc / B_trans was off by up to
# 1.9e-10 (1e-11 to 1.9e-10 with the seed at 0.3). References: the arbitrary-precision ISEM route at 128 bits (checked
# at 160 bits to 4e-36; GSN_ISSUES_LIVE H-007).
@testset "Y_radial horizon seed near m Omega_H" begin
    references = (
        (0.8, -6.60793022180392274255e-02 - 7.17183047964322964848e-03im),
        (0.835, -2.07847821956810119113e-02 + 1.68311756463033434620e-02im),
    )
    for (omega, reference) in references
        Y = Y_radial(-2, 2, 2, 0.99, omega, IN)
        @test Y.incidence_amplitude / Y.transmission_amplitude ≈ reference rtol=5e-11
    end
end
