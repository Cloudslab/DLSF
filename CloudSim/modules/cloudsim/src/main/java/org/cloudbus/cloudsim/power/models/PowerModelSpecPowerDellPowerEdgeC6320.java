package org.cloudbus.cloudsim.power.models;

/**
 * The power model of an Dell Inc. PowerEdge C6320 (Intel Xeon E5-2699 v3 2.30 GHz)
 * *The data is accessed from https://www.spec.org/power_ssj2008/results/res2015q3/power_ssj2008-20150707-00697.html
 * @author Shashikant Ilager
 * @since CloudSim Toolkit 3.0
 *
 * */

public class PowerModelSpecPowerDellPowerEdgeC6320  extends PowerModelSpecPower {

    /** The power.
     */
    private final double[] power = { 210, 371, 449, 522, 589, 647, 705, 802, 924, 1071, 1229};

    /*
     * (non-Javadoc)
     * @see org.cloudbus.cloudsim.power.models.PowerModelSpecPower#getPowerData(int)
     */
    @Override
    protected double getPowerData(int index) {
        return power[index];
    }
}