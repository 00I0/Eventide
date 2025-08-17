#pragma once
/**
 * @file TrajectoryResult.h
 * @brief Possible outcomes of a simulated trajectory.
 */

/**
 * @brief Codes returned by Simulator::processTrajectory.
 */
namespace eventide {
    enum class TrajectoryResult: int { ACCEPTED, REJECTED, CAPPED_AT_MAX_CASES, CAPPED_AT_T_RUN };
}
