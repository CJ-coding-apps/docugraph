CREATE NODE TABLE `Entity` (`uuid` STRING,`name` STRING,`group_id` STRING,`labels` STRING[],`created_at` TIMESTAMP,`name_embedding` FLOAT[],`summary` STRING,`attributes` STRING, PRIMARY KEY(`uuid`));
CREATE NODE TABLE `Community` (`uuid` STRING,`name` STRING,`group_id` STRING,`created_at` TIMESTAMP,`name_embedding` FLOAT[],`summary` STRING, PRIMARY KEY(`uuid`));
CREATE NODE TABLE `RelatesToNode_` (`uuid` STRING,`group_id` STRING,`created_at` TIMESTAMP,`name` STRING,`fact` STRING,`fact_embedding` FLOAT[],`episodes` STRING[],`expired_at` TIMESTAMP,`valid_at` TIMESTAMP,`invalid_at` TIMESTAMP,`reference_time` TIMESTAMP,`attributes` STRING, PRIMARY KEY(`uuid`));
CREATE NODE TABLE `Episodic` (`uuid` STRING,`name` STRING,`group_id` STRING,`created_at` TIMESTAMP,`source` STRING,`source_description` STRING,`content` STRING,`valid_at` TIMESTAMP,`entity_edges` STRING[], PRIMARY KEY(`uuid`));
CREATE NODE TABLE `Saga` (`uuid` STRING,`name` STRING,`group_id` STRING,`created_at` TIMESTAMP, PRIMARY KEY(`uuid`));
CREATE REL TABLE `HAS_MEMBER` (FROM `Community` TO `Entity`, FROM `Community` TO `Community`, `uuid` STRING,`group_id` STRING,`created_at` TIMESTAMP,MANY_MANY);
CREATE REL TABLE `RELATES_TO` (FROM `Entity` TO `RelatesToNode_`, FROM `RelatesToNode_` TO `Entity`, MANY_MANY);
CREATE REL TABLE `MENTIONS` (FROM `Episodic` TO `Entity`, `uuid` STRING,`group_id` STRING,`created_at` TIMESTAMP,MANY_MANY);
CREATE REL TABLE `HAS_EPISODE` (FROM `Saga` TO `Episodic`, `uuid` STRING,`group_id` STRING,`created_at` TIMESTAMP,MANY_MANY);
CREATE REL TABLE `NEXT_EPISODE` (FROM `Episodic` TO `Episodic`, `uuid` STRING,`group_id` STRING,`created_at` TIMESTAMP,MANY_MANY);
CREATE MACRO `2_COMMUNITY_NAME_TOKENIZE` (query) AS string_split(lower(regexp_replace(
                            CAST(query as STRING),
                            '[0-9!@#$%^&*()_+={}\\[\\]:;<>,.?~\\/\\|\'"`-]+',
                            ' ',
                            'g')), ' ');
CREATE MACRO `3_EDGE_NAME_AND_FACT_TOKENIZE` (query) AS string_split(lower(regexp_replace(
                            CAST(query as STRING),
                            '[0-9!@#$%^&*()_+={}\\[\\]:;<>,.?~\\/\\|\'"`-]+',
                            ' ',
                            'g')), ' ');
CREATE MACRO `0_EPISODE_CONTENT_TOKENIZE` (query) AS string_split(lower(regexp_replace(
                            CAST(query as STRING),
                            '[0-9!@#$%^&*()_+={}\\[\\]:;<>,.?~\\/\\|\'"`-]+',
                            ' ',
                            'g')), ' ');
CREATE MACRO `1_NODE_NAME_AND_SUMMARY_TOKENIZE` (query) AS string_split(lower(regexp_replace(
                            CAST(query as STRING),
                            '[0-9!@#$%^&*()_+={}\\[\\]:;<>,.?~\\/\\|\'"`-]+',
                            ' ',
                            'g')), ' ');
