package org.cloudbus.cloudsim.provisioners;

import org.cloudbus.cloudsim.Vm;

/**
 * DiskBwProvisioner is an abstract class that represents the provisioning policy used by a host
 * to allocate bandwidth (DiskBw) to virtual machines inside it. 
 * Each host has to have its own instance of a DiskDiskBwProvisioner.
 * When extending this class, care must be taken to guarantee that
 * the field availableDiskBw will always contain the amount of free bandwidth available for future
 * allocations.
 *
 * @author Shreshth Tuli
 * @since CloudSim Toolkit 1.0
 */
public abstract class DiskBwProvisioner {
    /** The total bandwidth capacity from the host that the provisioner can allocate to VMs. */
    private long DiskBw;

    /** The available bandwidth. */
    private long availableDiskBw;

    /**
     * Creates the new DiskBwProvisioner.
     *
     * @param DiskBw The total bandwidth capacity from the host that the provisioner can allocate to VMs.
     *
     * @pre DiskBw >= 0
     * @post $none
     */
    public DiskBwProvisioner(long DiskBw) {
        setDiskBw(DiskBw);
        setAvailableDiskBw(DiskBw);
    }

    /**
     * Allocates DiskBw for a given VM.
     *
     * @param vm the virtual machine for which the DiskBw are being allocated
     * @param DiskBw the DiskBw to be allocated to the VM
     *
     * @return $true if the DiskBw could be allocated; $false otherwise
     *
     * @pre $none
     * @post $none
     */
    public abstract boolean allocateDiskBwForVm(Vm vm, long DiskBw);

    /**
     * Gets the allocated DiskBw for VM.
     *
     * @param vm the VM
     *
     * @return the allocated DiskBw for vm
     */
    public abstract long getAllocatedDiskBwForVm(Vm vm);

    /**
     * Releases DiskBw used by a VM.
     *
     * @param vm the vm
     *
     * @pre $none
     * @post none
     */
    public abstract void deallocateDiskBwForVm(Vm vm);

    /**
     * Releases DiskBw used by all VMs.
     *
     * @pre $none
     * @post none
     */
    public void deallocateDiskBwForAllVms() {
        setAvailableDiskBw(getDiskBw());
    }

    /**
     * Checks if it is possible to change the current allocated DiskBw for the VM
     * to a new amount, depending on the available DiskBw.
     *
     * @param vm the vm to check if there is enough available DiskBw on the host to
     * change the VM allocated DiskBw
     * @param DiskBw the new total amount of DiskBw for the VM.
     *
     * @return true, if is suitable for vm
     */
    public abstract boolean isSuitableForVm(Vm vm, long DiskBw);

    /**
     * Gets the DiskBw capacity.
     *
     * @return the DiskBw capacity
     */
    public long getDiskBw() {
        return DiskBw;
    }

    /**
     * Sets the DiskBw capacity.
     *
     * @param DiskBw the new DiskBw capacity
     */
    protected void setDiskBw(long DiskBw) {
        this.DiskBw = DiskBw;
    }

    /**
     * Gets the available DiskBw in the host.
     *
     * @return available DiskBw
     *
     * @pre $none
     * @post $none
     */
    public long getAvailableDiskBw() {
        return availableDiskBw;
    }

    /**
     * Gets the amount of used DiskBw in the host.
     *
     * @return used DiskBw
     *
     * @pre $none
     * @post $none
     */
    public long getUsedDiskBw() {
        return DiskBw - availableDiskBw;
    }

    /**
     * Sets the available DiskBw.
     *
     * @param availableDiskBw the new available DiskBw
     */
    protected void setAvailableDiskBw(long availableDiskBw) {
        this.availableDiskBw = availableDiskBw;
    }
}
