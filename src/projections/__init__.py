"""
Projection service for maintaining Neo4j read models from EventStoreDB.

The projection service consumes events from ESDB via persistent subscriptions
and updates Neo4j in a partitioned, ordered manner to maintain consistency.
"""
