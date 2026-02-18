import { useDeferredValue, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Label } from "@/components/ui/label";
import {
  useEndpoints,
  type EndpointOut,
  type EndpointCategory,
} from "@/lib/benchmark-api";
import { Button } from "@/components/ui/button";
import { Search, Server } from "lucide-react";

const CATEGORY_ORDER: EndpointCategory[] = [
  "PAY_PER_TOKEN",
  "PROVISIONED_THROUGHPUT",
  "EXTERNAL_MODEL",
  "CUSTOM",
];

const CATEGORY_META: Record<
  EndpointCategory,
  { label: string; borderClass: string; badgeBg: string; badgeText: string }
> = {
  PAY_PER_TOKEN: {
    label: "Pay-per-token",
    borderClass: "border-l-blue-500",
    badgeBg: "bg-blue-100 dark:bg-blue-950",
    badgeText: "text-blue-700 dark:text-blue-300",
  },
  PROVISIONED_THROUGHPUT: {
    label: "Provisioned Throughput",
    borderClass: "border-l-violet-500",
    badgeBg: "bg-violet-100 dark:bg-violet-950",
    badgeText: "text-violet-700 dark:text-violet-300",
  },
  EXTERNAL_MODEL: {
    label: "External Model",
    borderClass: "border-l-amber-500",
    badgeBg: "bg-amber-100 dark:bg-amber-950",
    badgeText: "text-amber-700 dark:text-amber-300",
  },
  CUSTOM: {
    label: "Custom",
    borderClass: "border-l-gray-400",
    badgeBg: "bg-gray-100 dark:bg-gray-800",
    badgeText: "text-gray-600 dark:text-gray-400",
  },
};

interface EndpointSelectorProps {
  selected: string[];
  onSelectionChange: (selected: string[]) => void;
}

export function EndpointSelector({
  selected,
  onSelectionChange,
}: EndpointSelectorProps) {
  const [search, setSearch] = useState("");
  const [categoryFilter, setCategoryFilter] = useState<EndpointCategory | null>(null);
  const deferredFilter = useDeferredValue(categoryFilter);
  const deferredSearch = useDeferredValue(search);
  const { data: endpoints, isLoading, error } = useEndpoints();

  const filtered = useMemo(() => {
    const q = deferredSearch.toLowerCase();
    const matched = (endpoints ?? []).filter(
      (ep) =>
        (ep.name.toLowerCase().includes(q) ||
          (ep.model_name ?? "").toLowerCase().includes(q)) &&
        (deferredFilter === null || ep.endpoint_type === deferredFilter),
    );

    const order = Object.fromEntries(
      CATEGORY_ORDER.map((c, i) => [c, i]),
    ) as Record<string, number>;

    return matched.sort(
      (a, b) =>
        (order[a.endpoint_type] ?? 99) - (order[b.endpoint_type] ?? 99),
    );
  }, [endpoints, deferredSearch, deferredFilter]);

  const toggleEndpoint = (name: string) => {
    if (selected.includes(name)) {
      onSelectionChange(selected.filter((n) => n !== name));
    } else if (selected.length < 4) {
      onSelectionChange([...selected, name]);
    }
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Server className="h-4 w-4" />
          Select Endpoints
          <div className="ml-auto flex items-center gap-2">
            {selected.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                className="h-6 px-2 text-xs text-muted-foreground"
                onClick={() => onSelectionChange([])}
              >
                Deselect all
              </Button>
            )}
            <Badge variant="secondary">
              {selected.length}/4
            </Badge>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="relative">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search endpoints..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9 h-9"
          />
        </div>

        <CategoryLegend
          active={categoryFilter}
          onToggle={(cat) =>
            setCategoryFilter((prev) => (prev === cat ? null : cat))
          }
        />

        <div className="max-h-[280px] overflow-y-auto space-y-1 pr-1">
          {isLoading && (
            <div className="space-y-2">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          )}

          {error && (
            <p className="text-sm text-destructive py-2">
              Failed to load endpoints. Check your workspace connection.
            </p>
          )}

          {!isLoading && filtered.length === 0 && (
            <p className="text-sm text-muted-foreground py-4 text-center">
              {search ? "No endpoints match your search" : "No serving endpoints found"}
            </p>
          )}

          {filtered.map((ep) => (
            <EndpointRow
              key={ep.name}
              endpoint={ep}
              checked={selected.includes(ep.name)}
              disabled={!selected.includes(ep.name) && selected.length >= 4}
              onToggle={() => toggleEndpoint(ep.name)}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function CategoryLegend({
  active,
  onToggle,
}: {
  active: EndpointCategory | null;
  onToggle: (cat: EndpointCategory) => void;
}) {
  return (
    <div className="flex flex-wrap gap-x-1 gap-y-1 text-[10px]">
      {CATEGORY_ORDER.map((cat) => {
        const meta = CATEGORY_META[cat];
        const isActive = active === cat;
        return (
          <button
            key={cat}
            type="button"
            onClick={() => onToggle(cat)}
            className={`flex items-center gap-1 rounded-full px-2 py-0.5 cursor-pointer ${
              isActive
                ? `${meta.badgeBg} ${meta.badgeText} ring-1 ring-current`
                : "hover:bg-accent"
            }`}
          >
            <span
              className={`inline-block h-2 w-2 rounded-sm ${meta.badgeBg} border ${meta.borderClass}`}
            />
            <span className={isActive ? meta.badgeText : "text-muted-foreground"}>
              {meta.label}
            </span>
          </button>
        );
      })}
    </div>
  );
}

function EndpointRow({
  endpoint,
  checked,
  disabled,
  onToggle,
}: {
  endpoint: EndpointOut;
  checked: boolean;
  disabled: boolean;
  onToggle: () => void;
}) {
  const stateColor =
    endpoint.state === "READY"
      ? "bg-green-500"
      : endpoint.state === "NOT_READY"
        ? "bg-yellow-500"
        : "bg-gray-400";

  const cat = CATEGORY_META[endpoint.endpoint_type] ?? CATEGORY_META.CUSTOM;

  return (
    <Label
      className={`flex items-center gap-3 rounded-md border border-l-[3px] p-3 cursor-pointer transition-colors hover:bg-accent/50 ${cat.borderClass} ${
        checked ? "border-primary bg-accent/30 !border-l-primary" : ""
      } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
    >
      <Checkbox
        checked={checked}
        onCheckedChange={onToggle}
        disabled={disabled}
      />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium truncate">{endpoint.name}</span>
          <span
            className={`h-2 w-2 rounded-full shrink-0 ${stateColor}`}
            title={endpoint.state}
          />
        </div>
        <div className="flex items-center gap-2 mt-0.5">
          {endpoint.model_name && (
            <span className="text-xs text-muted-foreground truncate">
              {endpoint.model_name}
            </span>
          )}
          <Badge
            variant="outline"
            className={`text-[10px] h-4 px-1 border-0 ${cat.badgeBg} ${cat.badgeText}`}
          >
            {cat.label}
          </Badge>
        </div>
      </div>
    </Label>
  );
}
