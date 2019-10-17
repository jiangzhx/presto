package io.prestosql.sql.planner.optimizations;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import io.prestosql.Session;
import io.prestosql.execution.warnings.WarningCollector;
import io.prestosql.metadata.FunctionRegistry;
import io.prestosql.metadata.Signature;
import io.prestosql.operator.aggregation.InternalAggregationFunction;
import io.prestosql.sql.planner.Partitioning;
import io.prestosql.sql.planner.PartitioningScheme;
import io.prestosql.sql.planner.PlanNodeIdAllocator;
import io.prestosql.sql.planner.Symbol;
import io.prestosql.sql.planner.SymbolAllocator;
import io.prestosql.sql.planner.SymbolsExtractor;
import io.prestosql.sql.planner.TypeProvider;
import io.prestosql.sql.planner.plan.AggregationNode;
import io.prestosql.sql.planner.plan.AggregationNode.Aggregation;
import io.prestosql.sql.planner.plan.Assignments;
import io.prestosql.sql.planner.plan.ExchangeNode;
import io.prestosql.sql.planner.plan.JoinNode;
import io.prestosql.sql.planner.plan.JoinNode.EquiJoinClause;
import io.prestosql.sql.planner.plan.PlanNode;
import io.prestosql.sql.planner.plan.ProjectNode;
import io.prestosql.sql.planner.plan.SimplePlanRewriter;
import io.prestosql.sql.tree.FunctionCall;
import io.prestosql.sql.tree.QualifiedName;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.base.Verify.verify;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static io.prestosql.SystemSessionProperties.isPushAggregationThroughJoin;
import static io.prestosql.sql.planner.plan.AggregationNode.Step.FINAL;
import static io.prestosql.sql.planner.plan.AggregationNode.Step.PARTIAL;
import static io.prestosql.sql.planner.plan.AggregationNode.Step.SINGLE;
import static io.prestosql.sql.planner.plan.ExchangeNode.Type.GATHER;
import static io.prestosql.sql.planner.plan.ExchangeNode.Type.REPARTITION;
import static java.util.Objects.requireNonNull;

public class PartialAggregationPushDown
        implements PlanOptimizer
{
    private final FunctionRegistry functionRegistry;

    public PartialAggregationPushDown(FunctionRegistry registry)
    {
        requireNonNull(registry, "registry is null");

        this.functionRegistry = registry;
    }

    @Override
    public PlanNode optimize(PlanNode plan, Session session, TypeProvider types, SymbolAllocator symbolAllocator, PlanNodeIdAllocator idAllocator, WarningCollector warningCollector)
    {
        return SimplePlanRewriter.rewriteWith(new Rewriter(symbolAllocator, idAllocator, isPushAggregationThroughJoin(session)), plan, null);
    }

    private class Rewriter
            extends SimplePlanRewriter<Void>
    {
        private final SymbolAllocator allocator;
        private final PlanNodeIdAllocator idAllocator;
        private final boolean pushAggregationThroughJoin;

        public Rewriter(SymbolAllocator allocator, PlanNodeIdAllocator idAllocator, boolean pushAggregationThroughJoin)
        {
            this.allocator = requireNonNull(allocator, "allocator is null");
            this.idAllocator = requireNonNull(idAllocator, "idAllocator is null");
            this.pushAggregationThroughJoin = pushAggregationThroughJoin;
        }

        @Override
        public PlanNode visitAggregation(AggregationNode node, RewriteContext<Void> context)
        {
            PlanNode child = node.getSource();

            if (child instanceof JoinNode && pushAggregationThroughJoin) {
                return pushPartialThroughJoin(node, (JoinNode) child, context);
            }

            if (!(child instanceof ExchangeNode)) {
                return context.defaultRewrite(node);
            }

            boolean decomposable = node.isDecomposable(functionRegistry);

            if (node.getStep().equals(SINGLE) &&
                    node.hasEmptyGroupingSet() &&
                    node.hasNonEmptyGroupingSet()) {
                checkState(
                        decomposable,
                        "Distributed aggregation with empty grouping set requires partial but functions are not decomposable");
                return context.rewrite(split(node));
            }

            if (!decomposable) {
                return context.defaultRewrite(node);
            }

            // partial aggregation can only be pushed through exchange that doesn't change
            // the cardinality of the stream (i.e., gather or repartition)
            ExchangeNode exchange = (ExchangeNode) child;
            if ((exchange.getType() != GATHER && exchange.getType() != REPARTITION) ||
                    exchange.getPartitioningScheme().isReplicateNullsAndAny()) {
                return context.defaultRewrite(node);
            }

            if (exchange.getType() == REPARTITION) {
                // if partitioning columns are not a subset of grouping keys,
                // we can't push this through
                List<Symbol> partitioningColumns = exchange.getPartitioningScheme()
                        .getPartitioning()
                        .getArguments()
                        .stream()
                        .filter(Partitioning.ArgumentBinding::isVariable)
                        .map(Partitioning.ArgumentBinding::getColumn)
                        .collect(Collectors.toList());

                if (!node.getGroupingKeys().containsAll(partitioningColumns)) {
                    return context.defaultRewrite(node);
                }
            }

            // currently, we only support plans that don't use pre-computed hash functions
            if (node.getHashSymbol().isPresent() || exchange.getPartitioningScheme().getHashColumn().isPresent()) {
                return context.defaultRewrite(node);
            }

            switch (node.getStep()) {
                case SINGLE:
                    // Split it into a FINAL on top of a PARTIAL and
                    // reprocess the resulting plan to push the partial
                    // below the exchange (see case below).
                    return context.rewrite(split(node));
                case PARTIAL:
                    // Push it underneath each branch of the exchange
                    // and reprocess in case it can be pushed further down
                    // (e.g., if there are local/remote exchanges stacked)
                    return context.rewrite(pushPartial(node, exchange));
                default:
                    return context.defaultRewrite(node);
            }
        }

        private PlanNode pushPartialThroughJoin(AggregationNode node, JoinNode child, RewriteContext<Void> context)
        {
            if (node.getStep() != PARTIAL || node.getGroupingSets().getGroupingKeys().size() != 1) {
                return context.defaultRewrite(node);
            }

            if (child.getType() != JoinNode.Type.INNER || child.getFilter().isPresent()) {
                // TODO: add support for filter function.
                // All availableSymbols used in filter function could be added to pushedDownGroupingSet
                return context.defaultRewrite(node);
            }

            // TODO: leave partial aggregation above Join?
            if (allAggregationsOn(node.getAggregations(), child.getLeft().getOutputSymbols())) {
                return pushPartialToLeftChild(node, child, context);
            }
            else if (allAggregationsOn(node.getAggregations(), child.getRight().getOutputSymbols())) {
                return pushPartialToRightChild(node, child, context);
            }

            return context.defaultRewrite(node);
        }

        private PlanNode pushPartialToLeftChild(AggregationNode node, JoinNode child, RewriteContext<Void> context)
        {
            List<Symbol> groupingSet = getPushedDownGroupingSet(node, child, ImmutableSet.copyOf(child.getLeft().getOutputSymbols()));
            AggregationNode pushedAggregation = replaceAggregationSource(node, child.getLeft(), child.getCriteria(), groupingSet, context);
            return pushPartialToJoin(pushedAggregation, child, pushedAggregation, context.rewrite(child.getRight()), child.getRight().getOutputSymbols());
        }

        private PlanNode pushPartialToRightChild(AggregationNode node, JoinNode child, RewriteContext<Void> context)
        {
            List<Symbol> groupingSet = getPushedDownGroupingSet(node, child, ImmutableSet.copyOf(child.getRight().getOutputSymbols()));
            AggregationNode pushedAggregation = replaceAggregationSource(node, child.getRight(), child.getCriteria(), groupingSet, context);
            return pushPartialToJoin(pushedAggregation, child, context.rewrite(child.getLeft()), pushedAggregation, child.getLeft().getOutputSymbols());
        }

        private PlanNode pushPartialToJoin(
                AggregationNode pushedAggregation,
                JoinNode child,
                PlanNode leftChild,
                PlanNode rightChild,
                Collection<Symbol> otherSymbols)
        {
            ImmutableList.Builder<Symbol> outputSymbols = ImmutableList.builder();
            outputSymbols.addAll(pushedAggregation.getOutputSymbols());
            outputSymbols.addAll(otherSymbols);

            return new JoinNode(
                    child.getId(),
                    child.getType(),
                    leftChild,
                    rightChild,
                    child.getCriteria(),
                    outputSymbols.build(),
                    child.getFilter(),
                    child.getLeftHashSymbol(),
                    child.getRightHashSymbol(),
                    child.getDistributionType(),
                    Optional.empty());
        }

        private AggregationNode replaceAggregationSource(
                AggregationNode aggregation,
                PlanNode source,
                List<EquiJoinClause> criteria,
                List<Symbol> groupingSet,
                RewriteContext<Void> context)
        {
            PlanNode rewrittenSource = context.rewrite(source);
            ImmutableSet<Symbol> rewrittenSourceSymbols = ImmutableSet.copyOf(rewrittenSource.getOutputSymbols());
            ImmutableMap.Builder<Symbol, Symbol> mapping = ImmutableMap.builder();

            for (EquiJoinClause joinClause : criteria) {
                if (rewrittenSourceSymbols.contains(joinClause.getLeft())) {
                    mapping.put(joinClause.getRight(), joinClause.getLeft());
                }
                else {
                    mapping.put(joinClause.getLeft(), joinClause.getRight());
                }
            }

            AggregationNode pushedAggregation = new AggregationNode(
                    aggregation.getId(),
                    aggregation.getSource(),
                    aggregation.getAggregations(),
                    aggregation.getGroupingSets(),
                    aggregation.getPreGroupedSymbols(),
                    aggregation.getStep(),
                    aggregation.getHashSymbol(),
                    aggregation.getGroupIdSymbol());
            return new SymbolMapper(mapping.build()).map(pushedAggregation, source);
        }

        private boolean allAggregationsOn(Map<Symbol, Aggregation> aggregations, List<Symbol> outputSymbols)
        {
            Set<Symbol> inputs = SymbolsExtractor.extractUnique(aggregations.values().stream().map(Aggregation::getCall).collect(toImmutableList()));
            return outputSymbols.containsAll(inputs);
        }

        private List<Symbol> getPushedDownGroupingSet(AggregationNode aggregation, JoinNode join, Set<Symbol> availableSymbols)
        {
//            List<Symbol> groupingSet = Iterables.getOnlyElement(aggregation.getGroupingSets());
            List<Symbol> groupingSet = aggregation.getGroupingSets().getGroupingKeys();

            Set<Symbol> joinKeys = Stream.concat(
                    join.getCriteria().stream().map(EquiJoinClause::getLeft),
                    join.getCriteria().stream().map(EquiJoinClause::getRight)
            ).collect(Collectors.toSet());

            // keep symbols that are either directly from the join's child (availableSymbols) or there is
            // an equality in join condition to a symbol for the join child
            List<Symbol> pushedDownGroupingSet = groupingSet.stream()
                    .filter(symbol -> joinKeys.contains(symbol) || availableSymbols.contains(symbol))
                    .collect(Collectors.toList());

            if (pushedDownGroupingSet.size() != groupingSet.size() || pushedDownGroupingSet.isEmpty()) {
                // If we dropped some symbol, we have to add all join key columns to the grouping set
                Set<Symbol> existingSymbols = ImmutableSet.copyOf(pushedDownGroupingSet);

                join.getCriteria().stream()
                        .filter(equiJoinClause -> !existingSymbols.contains(equiJoinClause.getLeft()) && !existingSymbols.contains(equiJoinClause.getRight()))
                        .forEach(joinClause -> pushedDownGroupingSet.add(joinClause.getLeft()));
            }
            return pushedDownGroupingSet;
        }

        private PlanNode pushPartial(AggregationNode partial, ExchangeNode exchange)
        {
            List<PlanNode> partials = new ArrayList<>();
            for (int i = 0; i < exchange.getSources().size(); i++) {
                PlanNode source = exchange.getSources().get(i);

                SymbolMapper.Builder mappingsBuilder = SymbolMapper.builder();
                for (int outputIndex = 0; outputIndex < exchange.getOutputSymbols().size(); outputIndex++) {
                    Symbol output = exchange.getOutputSymbols().get(outputIndex);
                    Symbol input = exchange.getInputs().get(i).get(outputIndex);
                    if (!output.equals(input)) {
                        mappingsBuilder.put(output, input);
                    }
                }

                SymbolMapper symbolMapper = mappingsBuilder.build();
                AggregationNode mappedPartial = symbolMapper.map(partial, source, idAllocator);

                Assignments.Builder assignments = Assignments.builder();

                for (Symbol output : partial.getOutputSymbols()) {
                    Symbol input = symbolMapper.map(output);
                    assignments.put(output, input.toSymbolReference());
                }
                partials.add(new ProjectNode(idAllocator.getNextId(), mappedPartial, assignments.build()));
            }

            for (PlanNode node : partials) {
                verify(partial.getOutputSymbols().equals(node.getOutputSymbols()));
            }

            // Since this exchange source is now guaranteed to have the same symbols as the inputs to the the partial
            // aggregation, we don't need to rewrite symbols in the partitioning function
            PartitioningScheme partitioning = new PartitioningScheme(
                    exchange.getPartitioningScheme().getPartitioning(),
                    partial.getOutputSymbols(),
                    exchange.getPartitioningScheme().getHashColumn(),
                    exchange.getPartitioningScheme().isReplicateNullsAndAny(),
                    exchange.getPartitioningScheme().getBucketToPartition());

            return new ExchangeNode(
                    idAllocator.getNextId(),
                    exchange.getType(),
                    exchange.getScope(),
                    partitioning,
                    partials,
                    ImmutableList.copyOf(Collections.nCopies(partials.size(), partial.getOutputSymbols())),
                    Optional.empty());
        }

        private PlanNode split(AggregationNode node)
        {
            // otherwise, add a partial and final with an exchange in between
            Map<Symbol, Aggregation> intermediateAggregation = new HashMap<>();
            Map<Symbol, Aggregation> finalAggregation = new HashMap<>();
            for (Map.Entry<Symbol, Aggregation> entry : node.getAggregations().entrySet()) {
                Aggregation originalAggregation = entry.getValue();
                Signature signature = originalAggregation.getSignature();
                InternalAggregationFunction function = functionRegistry.getAggregateFunctionImplementation(signature);
                Symbol intermediateSymbol = allocator.newSymbol(signature.getName(), function.getIntermediateType());

                intermediateAggregation.put(intermediateSymbol, new Aggregation(originalAggregation.getCall(), signature, originalAggregation.getMask()));

                // rewrite final aggregation in terms of intermediate function
                finalAggregation.put(entry.getKey(),
                        new Aggregation(
                                new FunctionCall(QualifiedName.of(signature.getName()), ImmutableList.of(intermediateSymbol.toSymbolReference())),
                                signature,
                                Optional.empty()));
            }

            PlanNode partial = new AggregationNode(
                    idAllocator.getNextId(),
                    node.getSource(),
                    intermediateAggregation,
                    node.getGroupingSets(),
                    node.getPreGroupedSymbols(),
                    PARTIAL,
                    node.getHashSymbol(),
                    node.getGroupIdSymbol());

            return new AggregationNode(
                    node.getId(),
                    partial,
                    finalAggregation,
                    node.getGroupingSets(),
                    node.getPreGroupedSymbols(),
                    FINAL,
                    node.getHashSymbol(),
                    node.getGroupIdSymbol());
        }
    }
}
