package org.cloudbus.cloudsim.power.models;

/**
 * The power model of an Hitachi HA 8000 (Intel Core i3-540 2 core 3.06 GHz)
 * *The data is accessed from https://www.spec.org/power_ssj2008/results/res2011q1/power_ssj2008-20110206-00346.html
 * @author Shreshth Tuli
 * @since CloudSim Toolkit 5.0
 *
 * */

public class PowerModelSpecPowerHitachiHA8000 extends PowerModelSpecPower {

    /** The power.
     */
    private final double[] power = { 24.3, 30.4, 33.7, 36.6, 39.6, 42.2, 45.6, 51.8, 55.7, 60.8, 63.2};

    /*
     * (non-Javadoc)
     * @see org.cloudbus.cloudsim.power.models.PowerModelSpecPower#getPowerData(int)
     */
    @Override
    protected double getPowerData(int index) {
        return power[index];
    }
}