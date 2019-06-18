package org.cloudbus.cloudsim.plus.billing;

import org.cloudbus.cloudsim.Vm;

import java.math.BigDecimal;
import java.util.List;

/**
 * Defines the billing policy of a cloud provider with respect to pricing VMs.
 * <strong>NOTE!</strong> - the implementations of this interface are completely
 * independent from the pricing properties defined in the data centre objects.
 * 
 * @author nikolay.grozev
 * 
 */
public interface IVmBillingPolicy {

    /**
     * Returns the cost for the specified vms.
     * 
     * @param vms
     *            - the vms to bill. Must not be empty
     * @return the cost for the specified vms.
     */
    public BigDecimal bill(final List<? extends Vm> vms);

    /**
     * Returns the cost for the specified vms before a specified moment in time.
     * 
     * @param vms
     * @param before
     * @return
     */
    public BigDecimal bill(final List<? extends Vm> vms, double before);

    /**
     * Returns the next charging time.
     * 
     * @param vm
     *            - the VM to check for. Must not be null;
     * @return the next charging time or -1 if the charge time could not be
     *         estimated.
     */
    public double nexChargeTime(final Vm vm);

    /**
     * Returns a normalised cost of the VM for 1 minute. This operation is
     * optional.
     * 
     * @param vm
     *            - the VM to check for. Must not be null;
     * @return a normalised cost of the VM for 1 minute.
     */
    public BigDecimal normalisedCostPerMinute(final Vm vm);

}
