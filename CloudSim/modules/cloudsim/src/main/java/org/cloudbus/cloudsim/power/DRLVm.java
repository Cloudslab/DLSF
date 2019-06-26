package org.cloudbus.cloudsim.power;

import org.cloudbus.cloudsim.CloudletScheduler;
import org.cloudbus.cloudsim.core.CloudSim;

public class DRLVm extends PowerVm {

    public double totalResponseTime = 0;

    public double startTime = 0;

    public double totalMigrationTime = 0;

    public int delay = 0;

    /**
     * Instantiates a new PowerVm.
     *
     * @param id the id
     * @param userId the user id
     * @param mips the mips
     * @param pesNumber the pes number
     * @param ram the ram
     * @param bw the bw
     * @param size the size
     * @param priority the priority
     * @param vmm the vmm
     * @param cloudletScheduler the cloudlet scheduler
     * @param schedulingInterval the scheduling interval
     */
    public DRLVm(
            final int id,
            final int userId,
            final double mips,
            final int pesNumber,
            final int ram,
            final long bw,
            final long diskBw,
            final long size,
            final int priority,
            final String vmm,
            final CloudletScheduler cloudletScheduler,
            final double schedulingInterval,
            final int delay) {
        super(id, userId, mips, pesNumber, ram, bw, diskBw, size, priority, vmm, cloudletScheduler, schedulingInterval);
        this.startTime = CloudSim.clock();
        this.delay = delay;
    }
}
